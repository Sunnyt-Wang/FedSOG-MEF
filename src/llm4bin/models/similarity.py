from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import Linear
from torch_geometric.nn import SumAggregation
from torch_geometric.utils import softmax
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import Config
from .chat_template import build_prompt
from .gnn import LINEAR_INIT, EmbeddingEncoder, GGNN
from .graphormer import GraphormerSOGEncoder
from .lightweight import LightSOGEncoder
from .projector import SogProjector
from .similarity_loss import SimilarityLossOutput, gather_distributed_graph_vectors, hermessim_batch_pair_loss


SOG_RELATION_MODES = {
    "full",
    "no_side_effect",
    "no_data_dependency",
    "control_only",
    "none",
}


def filter_sog_relations(
    edge_index: torch.Tensor,
    edge_feat: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Filter SOG edges while retaining each selected edge's position code."""
    if mode not in SOG_RELATION_MODES:
        raise ValueError(
            f"unsupported model.graph_relation_mode={mode!r}; "
            f"expected one of {sorted(SOG_RELATION_MODES)}"
        )
    if mode == "full":
        return edge_index, edge_feat
    if mode == "none":
        return edge_index[:, :0], edge_feat[:0]

    # Preprocessing encodes 4 * position + one-based relation type; collate.py
    # subtracts one, so modulo four recovers the zero-based relation type.
    relation = torch.remainder(edge_feat.long(), 4)
    if mode == "no_side_effect":
        keep = relation < 2
    elif mode == "no_data_dependency":
        keep = relation != 0
    else:  # control_only
        keep = relation == 1
    return edge_index[:, keep], edge_feat[keep]


class GraphTokenPooler(nn.Module):
    def __init__(self, embed_dim: int, num_tokens: int):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_tokens, embed_dim) * 0.02)

    def forward(self, node_embeddings: torch.Tensor, graph_idx: torch.Tensor) -> torch.Tensor:
        batch_size = int(graph_idx.max().item()) + 1
        tokens = []
        for i in range(batch_size):
            nodes = node_embeddings[graph_idx == i]
            scores = nodes @ self.queries.transpose(0, 1)
            weights = torch.softmax(scores.transpose(0, 1), dim=-1)
            tokens.append(weights @ nodes)
        return torch.stack(tokens, dim=0)


class MSoftV2GraphAggregator(nn.Module):
    """HermesSim ``msoftv2`` graph aggregator for direct graph vectors."""

    def __init__(
        self,
        node_channels: int,
        num_queries: int,
        hidden_channels: int,
        n_node_trans: int,
        n_agg_trans: int,
        q_scale: float,
    ):
        super().__init__()
        if n_node_trans != 1 or n_agg_trans != 1:
            raise ValueError("msoftv2 currently supports HermesSim's n_node_trans=1 and n_agg_trans=1")
        multi_hidden = hidden_channels * num_queries
        self.nlin = nn.Sequential(Linear(node_channels, multi_hidden, **LINEAR_INIT), nn.ReLU(inplace=True))
        self.agglin = Linear(multi_hidden, multi_hidden, **LINEAR_INIT)
        self.l_norm = nn.LayerNorm(hidden_channels, elementwise_affine=False)
        self.queries = nn.Parameter(torch.full((num_queries, hidden_channels), float(q_scale)))
        self.sum_agg = SumAggregation()
        self.num_queries = num_queries
        self.hidden_channels = hidden_channels

    def forward(self, node_embeddings: torch.Tensor, graph_idx: torch.Tensor) -> torch.Tensor:
        batch_size = int(graph_idx.max().item()) + 1
        x = self.nlin(node_embeddings).reshape(-1, self.num_queries, self.hidden_channels)
        q = self.l_norm(x) * self.queries.unsqueeze(0)
        weights = softmax(q, graph_idx, num_nodes=batch_size, dim=0)
        pooled = self.sum_agg(weights * x, graph_idx, dim_size=batch_size, dim=0)
        return self.agglin(pooled.reshape(batch_size, -1))


class SimilarityGraphLLM(nn.Module):
    """Qwen-conditioned graph embedding model trained with HermesSim loss."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.arch = cfg.model.similarity_arch
        if self.arch not in {"qwen_fusion", "gnn_msoftv2", "graphormer_sog", "light_sog"}:
            raise ValueError(f"unsupported model.similarity_arch={self.arch!r}")
        if self.arch == "qwen_fusion":
            tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = "<|endoftext|>"
            tokenizer.add_special_tokens({"additional_special_tokens": [cfg.model.image_token]})
            self.tokenizer = tokenizer
            self.image_token = cfg.model.image_token
            self.image_token_id = tokenizer.convert_tokens_to_ids(cfg.model.image_token)
            self.pad_token_id = tokenizer.pad_token_id

            llm = AutoModelForCausalLM.from_pretrained(cfg.model.base_model_path, dtype=torch.bfloat16)
            llm.resize_token_embeddings(len(tokenizer), mean_resizing=True)
            if cfg.model.gradient_checkpointing:
                llm.gradient_checkpointing_enable()
                llm.enable_input_require_grads()
            if cfg.model.use_lora:
                for p in llm.parameters():
                    p.requires_grad = False
                lora_cfg = LoraConfig(
                    r=cfg.model.lora_r,
                    lora_alpha=cfg.model.lora_alpha,
                    target_modules=cfg.model.lora_target_modules,
                    lora_dropout=cfg.model.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                llm = get_peft_model(llm, lora_cfg)
                self._registered_llm = llm
            elif cfg.model.freeze_llm:
                for p in llm.parameters():
                    p.requires_grad = False
                llm.eval()
                object.__setattr__(self, "_frozen_llm", llm)
            else:
                self._registered_llm = llm
            if llm.config.hidden_size != cfg.model.hidden_size:
                raise ValueError(
                    f"cfg.model.hidden_size={cfg.model.hidden_size} disagrees with loaded "
                    f"model hidden_size={llm.config.hidden_size}"
                )

        if self.arch == "graphormer_sog":
            self.graphormer = GraphormerSOGEncoder(
                n_node_attr=cfg.ggnn.encoder.n_node_attr,
                n_edge_attr=cfg.ggnn.encoder.n_edge_attr,
                hidden_size=cfg.model.graphormer_hidden_size,
                num_layers=cfg.model.graphormer_num_layers,
                num_heads=cfg.model.graphormer_num_heads,
                ffn_dim=cfg.model.graphormer_ffn_dim,
                dropout=cfg.model.graphormer_dropout,
                max_nodes=cfg.model.graphormer_max_nodes,
                max_degree=cfg.model.graphormer_max_degree,
                max_distance=cfg.model.graphormer_max_distance,
                output_dim=cfg.model.similarity_dim,
            )
        elif self.arch == "light_sog":
            self.light_encoder = LightSOGEncoder(
                n_node_attr=cfg.ggnn.encoder.n_node_attr,
                n_edge_attr=cfg.ggnn.encoder.n_edge_attr,
                hidden_size=cfg.model.light_hidden_size,
                edge_hidden_size=cfg.model.light_edge_hidden_size,
                num_layers=cfg.model.light_num_layers,
                dropout=cfg.model.light_dropout,
                output_dim=cfg.model.similarity_dim,
            )
        else:
            encoder = EmbeddingEncoder(
                n_node_feat_dim=cfg.ggnn.encoder.n_node_feat_dim,
                n_edge_feat_dim=cfg.ggnn.encoder.n_edge_feat_dim,
                n_node_attr=cfg.ggnn.encoder.n_node_attr,
                n_edge_attr=cfg.ggnn.encoder.n_edge_attr,
                n_pos_enc=cfg.ggnn.encoder.n_pos_enc,
            )
            self.gnn = GGNN(
                encoder=encoder,
                n_node_feat_dim=cfg.ggnn.args.n_node_feat_dim,
                n_edge_feat_dim=cfg.ggnn.args.n_edge_feat_dim,
                layer_groups=cfg.ggnn.args.layer_groups,
                n_message_net_layers=cfg.ggnn.args.n_message_net_layers,
                skip_mode=cfg.ggnn.args.skip_mode,
                output_mode=cfg.ggnn.args.output_mode,
                num_query=cfg.ggnn.args.num_query,
                n_atte_layers=cfg.ggnn.args.n_atte_layers,
                layer_aggr=cfg.ggnn.args.layer_aggr,
                layer_aggr_kwargs=cfg.ggnn.args.layer_aggr_kwargs,
                concat_skip=cfg.ggnn.args.concat_skip,
            )
        if self.arch == "qwen_fusion":
            self.pooler = GraphTokenPooler(cfg.model.embed_dim, cfg.model.block_num)
            self.projector = SogProjector(embed_dim=cfg.model.embed_dim, hidden_size=cfg.model.hidden_size)
            self.similarity_head = nn.Linear(cfg.model.hidden_size, cfg.model.similarity_dim, bias=False)
        elif self.arch == "gnn_msoftv2":
            self.aggr = MSoftV2GraphAggregator(
                node_channels=cfg.ggnn.args.n_node_feat_dim,
                num_queries=cfg.model.msoft_num_queries,
                hidden_channels=cfg.model.msoft_hidden_channels,
                n_node_trans=cfg.model.msoft_node_trans,
                n_agg_trans=cfg.model.msoft_aggr_trans,
                q_scale=cfg.model.msoft_q_scale,
            )

    @property
    def llm(self) -> nn.Module:
        if self.arch != "qwen_fusion":
            raise RuntimeError("llm is only available for model.similarity_arch=qwen_fusion")
        frozen_llm = self.__dict__.get("_frozen_llm")
        if frozen_llm is not None:
            return frozen_llm
        return self._registered_llm

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def to(self, *args, **kwargs):
        module = super().to(*args, **kwargs)
        frozen_llm = self.__dict__.get("_frozen_llm")
        if frozen_llm is not None:
            frozen_llm.to(*args, **kwargs)
        return module

    def train(self, mode: bool = True):
        module = super().train(mode)
        frozen_llm = self.__dict__.get("_frozen_llm")
        if frozen_llm is not None:
            frozen_llm.eval()
        return module

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def state_dict(self, *args, destination=None, prefix: str = "", keep_vars: bool = False):
        sd: dict[str, torch.Tensor] = {} if destination is None else destination
        if self.arch == "graphormer_sog":
            self.graphormer.state_dict(destination=sd, prefix=f"{prefix}graphormer.", keep_vars=keep_vars)
            return sd
        if self.arch == "light_sog":
            self.light_encoder.state_dict(destination=sd, prefix=f"{prefix}light_encoder.", keep_vars=keep_vars)
            return sd
        self.gnn.state_dict(destination=sd, prefix=f"{prefix}gnn.", keep_vars=keep_vars)
        if self.arch == "qwen_fusion":
            self.pooler.state_dict(destination=sd, prefix=f"{prefix}pooler.", keep_vars=keep_vars)
            self.projector.state_dict(destination=sd, prefix=f"{prefix}projector.", keep_vars=keep_vars)
            self.similarity_head.state_dict(destination=sd, prefix=f"{prefix}similarity_head.", keep_vars=keep_vars)
            if self.cfg.model.use_lora:
                for name, param in self.llm.named_parameters():
                    if param.requires_grad:
                        value = param if keep_vars else param.detach()
                        sd[f"{prefix}_registered_llm.{name}"] = value
        else:
            self.aggr.state_dict(destination=sd, prefix=f"{prefix}aggr.", keep_vars=keep_vars)
        return sd

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        return super().load_state_dict(state_dict, strict=False, assign=assign)

    def _graph_tokens(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        node_feat = batch["node_feat"].to(self.device)
        edge_index = batch["edge_index"].to(self.device)
        edge_feat = batch["edge_feat"].to(self.device)
        graph_idx = batch["graph_idx"].to(self.device)
        relation_mode = "none" if self.cfg.model.disable_graph_relations else self.cfg.model.graph_relation_mode
        edge_index, edge_feat = filter_sog_relations(edge_index, edge_feat, relation_mode)
        node_embeddings, graph_idx = self.gnn(node_feat, edge_index, edge_feat, graph_idx)
        pooled = self.pooler(node_embeddings, graph_idx)
        return self.projector(pooled).to(dtype=self.llm.dtype)

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.arch == "graphormer_sog":
            node_feat = batch["node_feat"].to(self.device)
            edge_index = batch["edge_index"].to(self.device)
            edge_feat = batch["edge_feat"].to(self.device)
            graph_idx = batch["graph_idx"].to(self.device)
            relation_mode = "none" if self.cfg.model.disable_graph_relations else self.cfg.model.graph_relation_mode
            edge_index, edge_feat = filter_sog_relations(edge_index, edge_feat, relation_mode)
            return self.graphormer(node_feat, edge_index, edge_feat, graph_idx)

        if self.arch == "light_sog":
            node_feat = batch["node_feat"].to(self.device)
            edge_index = batch["edge_index"].to(self.device)
            edge_feat = batch["edge_feat"].to(self.device)
            graph_idx = batch["graph_idx"].to(self.device)
            relation_mode = "none" if self.cfg.model.disable_graph_relations else self.cfg.model.graph_relation_mode
            edge_index, edge_feat = filter_sog_relations(edge_index, edge_feat, relation_mode)
            return self.light_encoder(node_feat, edge_index, edge_feat, graph_idx)

        if self.arch == "gnn_msoftv2":
            node_feat = batch["node_feat"].to(self.device)
            edge_index = batch["edge_index"].to(self.device)
            edge_feat = batch["edge_feat"].to(self.device)
            graph_idx = batch["graph_idx"].to(self.device)
            relation_mode = "none" if self.cfg.model.disable_graph_relations else self.cfg.model.graph_relation_mode
            edge_index, edge_feat = filter_sog_relations(edge_index, edge_feat, relation_mode)
            node_embeddings, graph_idx = self.gnn(node_feat, edge_index, edge_feat, graph_idx)
            return self.aggr(node_embeddings, graph_idx)

        graph_tokens = self._graph_tokens(batch)
        bsz, num_tokens, _ = graph_tokens.shape
        prompt = build_prompt(
            self.tokenizer,
            self.cfg.prompt.system,
            self.cfg.prompt.user_prefix,
            self.image_token,
            num_tokens,
        )
        prompt_ids = self.tokenizer(
            [prompt] * bsz,
            return_tensors="pt",
            add_special_tokens=False,
            padding=False,
        ).input_ids.to(self.device)
        text_emb = self.llm.get_input_embeddings()(prompt_ids)
        image_positions = (prompt_ids[0] == self.image_token_id).nonzero(as_tuple=True)[0]
        if image_positions.numel() != num_tokens:
            raise RuntimeError(f"expected {num_tokens} image positions, got {image_positions.numel()}")
        inputs_embeds = text_emb.clone()
        inputs_embeds[:, image_positions, :] = graph_tokens
        attention_mask = (prompt_ids != self.pad_token_id).long()

        base_model = getattr(self.llm, "model", None)
        if self.cfg.model.use_lora or base_model is None:
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            hidden = outputs.hidden_states[-1]
        else:
            outputs = base_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
            )
            hidden = outputs.last_hidden_state
        pooled = hidden[:, -1, :]
        return self.similarity_head(pooled.float())

    def forward(self, batch: dict[str, torch.Tensor]) -> SimilarityLossOutput:
        graph_vectors = self.encode(batch)
        if self.cfg.train.gather_distributed_loss:
            graph_vectors = gather_distributed_graph_vectors(graph_vectors)
        return hermessim_batch_pair_loss(
            graph_vectors=graph_vectors,
            group_size=self.cfg.train.n_sim_funcs,
            margin=self.cfg.train.loss_margin,
            gama=self.cfg.train.loss_gama,
            norm_neg_sampling_s=self.cfg.train.norm_neg_sampling_s,
            graph_vec_regularizer_weight=self.cfg.train.graph_vec_regularizer_weight,
        )
