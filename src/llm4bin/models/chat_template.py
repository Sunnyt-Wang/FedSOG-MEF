from __future__ import annotations

from typing import Any


def build_messages(system: str, user_prefix: str, image_token: str, num_image_tokens: int) -> list[dict[str, str]]:
    """Build a Qwen3-style ``messages`` list for ``apply_chat_template``.

    The user message ends with ``num_image_tokens`` copies of ``image_token``;
    those positions are later overwritten with the projected GNN embeddings.
    """
    user_content = f"{user_prefix}{image_token * num_image_tokens}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def build_prompt(tokenizer: Any, system: str, user_prefix: str, image_token: str, num_image_tokens: int) -> str:
    msgs = build_messages(system, user_prefix, image_token, num_image_tokens)
    return tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
