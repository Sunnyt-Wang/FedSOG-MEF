"""GraphLLM: GGNN + projector + frozen LLM, glued for training and inference.

Design notes (delta from the legacy ``models/llama_model.py``):

* One ``build_inputs`` that handles both vuln (target='<digit>') and source
  (target=arbitrary text). Three near-duplicate methods in the legacy code are
  merged into this single path.
* The ``<image>`` placeholder is added with ``mean_resizing=True`` (legacy was
  ``False`` — empirically much worse for frozen LLM).
* Image-slot replacement uses a single vectorised ``in-place`` assignment on a
  cloned tensor instead of two nested Python loops.
* The LLM is frozen (``requires_grad=False``); we never call ``.eval()`` on it
  during training (the original code had ``.eval()`` in the training loop —
  see ``trainer.py:72`` in archived legacy).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import Config
from .chat_template import build_prompt
from .gnn import EmbeddingEncoder, GGNN
from .projector import SogProjector


@dataclass
class BuildInputsOutput:
    inputs_embeds: torch.Tensor          # [B, L, H]
    attention_mask: torch.Tensor         # [B, L]
    labels: torch.Tensor | None          # [B, L] or None
    prompt_len: int


class GraphLLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # --- Tokenizer ---
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_model_path)
        # Use Qwen3 endoftext as pad (distinct from eos=<|im_end|>).
        if tokenizer.pad_token is None:
            tokenizer.pad_token = "<|endoftext|>"
        # Add <image> as a special token.
        added = tokenizer.add_special_tokens({"additional_special_tokens": [cfg.model.image_token]})
        self.tokenizer = tokenizer
        self.image_token = cfg.model.image_token
        self.image_token_id = tokenizer.convert_tokens_to_ids(cfg.model.image_token)
        self.pad_token_id = tokenizer.pad_token_id
        assert self.image_token_id != tokenizer.unk_token_id, "image_token not registered"
        assert self.pad_token_id != tokenizer.eos_token_id, "pad must differ from eos"

        # --- LLM ---
        llm = AutoModelForCausalLM.from_pretrained(cfg.model.base_model_path, dtype=torch.bfloat16)
        llm.resize_token_embeddings(len(tokenizer), mean_resizing=True)
        if cfg.model.gradient_checkpointing:
            llm.gradient_checkpointing_enable()
            llm.enable_input_require_grads()
        if cfg.model.freeze_llm:
            for p in llm.parameters():
                p.requires_grad = False
        self.llm = llm

        hidden_size = llm.config.hidden_size
        if cfg.model.hidden_size != hidden_size:
            raise ValueError(
                f"cfg.model.hidden_size={cfg.model.hidden_size} disagrees with "
                f"loaded model hidden_size={hidden_size}"
            )
        self.hidden_size = hidden_size

        # --- GNN + projector ---
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
        self.projector = SogProjector(embed_dim=cfg.model.embed_dim, hidden_size=hidden_size)

        # --- Validate digit tokenization (vuln task) ---
        if cfg.train.task == "vuln":
            for i in range(cfg.prompt.num_classes):
                ids = tokenizer(str(i), add_special_tokens=False)["input_ids"]
                assert len(ids) == 1, f"digit {i} tokenizes to {ids}, expected one token"

    # -- Helpers ----------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return next(self.llm.parameters()).device

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    # Only persist GNN + projector. The frozen LLM weights are identical to
    # ``cfg.model.base_model_path`` and are reloaded from there on resume.
    def state_dict(self, *args, destination=None, prefix: str = "", keep_vars: bool = False):
        sd: dict[str, torch.Tensor] = {} if destination is None else destination
        self.gnn.state_dict(destination=sd, prefix=f"{prefix}gnn.", keep_vars=keep_vars)
        self.projector.state_dict(destination=sd, prefix=f"{prefix}projector.", keep_vars=keep_vars)
        return sd

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        # Always non-strict: state files contain only trainable keys (no LLM).
        return super().load_state_dict(state_dict, strict=False, assign=assign)

    # -- Core: encode graphs and merge with text embeddings --------------

    def _encode_graphs(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, list[int], int]:
        """Run GNN and return (projected [B, K, H], per-sample sizes, K)."""
        node_feat = batch["node_feat"].to(self.device)
        edge_index = batch["edge_index"].to(self.device)
        edge_feat = batch["edge_feat"].to(self.device)
        graph_idx = batch["graph_idx"].to(self.device)

        x, gi = self.gnn(node_feat, edge_index, edge_feat, graph_idx)
        # x is fp32 GNN output; project to hidden_size, then cast to LLM dtype.
        B = int(gi.max().item()) + 1
        per_sample = [x[gi == i] for i in range(B)]
        n_i = [t.shape[0] for t in per_sample]
        K = max(n_i)
        assert K > 0, "empty graphs in batch"
        # Stack -> [B, K, embed_dim]
        image_feat = x.new_zeros(B, K, x.shape[1])
        for i, t in enumerate(per_sample):
            image_feat[i, : n_i[i]] = t
        projected = self.projector(image_feat)
        projected = projected.to(dtype=self.llm.dtype)
        return projected, n_i, K

    def build_inputs(
        self,
        batch: dict[str, torch.Tensor],
        target_text: list[str] | None,
    ) -> BuildInputsOutput:
        device = self.device
        projected, n_i, K = self._encode_graphs(batch)
        B = projected.shape[0]

        # --- Prompt (identical across batch) ---
        prompt_str = build_prompt(
            self.tokenizer,
            self.cfg.prompt.system,
            self.cfg.prompt.user_prefix,
            self.image_token,
            K,
        )
        prompt_ids = self.tokenizer(
            [prompt_str] * B,
            return_tensors="pt",
            add_special_tokens=False,
            padding=False,
        ).input_ids.to(device)
        Lp = prompt_ids.shape[1]

        # --- Append target (training) ---
        target_attention_mask: torch.Tensor | None = None
        if target_text is not None:
            tgt = self.tokenizer(
                target_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.cfg.prompt.max_target_length,
                add_special_tokens=False,
            ).to(device)
            input_ids = torch.cat([prompt_ids, tgt.input_ids], dim=1)
            target_attention_mask = tgt.attention_mask
        else:
            input_ids = prompt_ids

        # --- Text embeddings (frozen) ---
        text_emb = self.llm.get_input_embeddings()(input_ids)  # [B, L, H]

        # --- Replace <image> slots with projected GNN features ---
        # prompt is identical across batch → image positions identical across batch.
        image_positions = (prompt_ids[0] == self.image_token_id).nonzero(as_tuple=True)[0]
        assert image_positions.numel() == K, f"expected {K} image positions, got {image_positions.numel()}"

        slot_mask_1d = torch.zeros(input_ids.shape[1], dtype=torch.bool, device=device)
        slot_mask_1d[image_positions] = True
        slot_mask = slot_mask_1d.unsqueeze(0).expand(B, -1)  # [B, L]
        padded_proj = torch.zeros_like(text_emb)
        padded_proj[:, image_positions, :] = projected  # [B, K, H] -> scattered into [B, L, H]
        inputs_embeds = torch.where(slot_mask.unsqueeze(-1), padded_proj, text_emb)

        # --- Attention mask ---
        attention_mask = (input_ids != self.pad_token_id).long()
        # Zero-out the trailing (K - n_i) image positions per sample (those have no real graph node).
        if not all(n == K for n in n_i):
            invalid_image = torch.zeros(B, K, dtype=torch.bool, device=device)
            for i in range(B):
                if n_i[i] < K:
                    invalid_image[i, n_i[i] :] = True
            attn_invalid = torch.zeros_like(attention_mask, dtype=torch.bool)
            attn_invalid[:, image_positions] = invalid_image
            attention_mask = attention_mask.masked_fill(attn_invalid, 0)

        # --- Labels (training only) ---
        labels: torch.Tensor | None = None
        if target_text is not None:
            labels = input_ids.clone()
            labels[:, :Lp] = -100
            if target_attention_mask is not None:
                tail = labels[:, Lp:].masked_fill(target_attention_mask == 0, -100)
                labels = torch.cat([labels[:, :Lp], tail], dim=1)

        return BuildInputsOutput(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, prompt_len=Lp)

    # -- Training / inference forward -----------------------------------

    def forward(self, batch: dict[str, torch.Tensor], target_text: list[str]) -> torch.Tensor:
        out = self.build_inputs(batch, target_text=target_text)
        result = self.llm(
            inputs_embeds=out.inputs_embeds,
            attention_mask=out.attention_mask,
            labels=out.labels,
        )
        return result.loss

    @torch.no_grad()
    def generate_step(self, batch: dict[str, torch.Tensor], max_new_tokens: int = 1) -> torch.Tensor:
        """Return ``[B, max_new_tokens]`` of generated token ids."""
        out = self.build_inputs(batch, target_text=None)
        gen = self.llm.generate(
            inputs_embeds=out.inputs_embeds,
            attention_mask=out.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        # With inputs_embeds, generate() returns only the new tokens (HF ≥4.34).
        return gen
