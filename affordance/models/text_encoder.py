from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class MiniLMTextEncoder(nn.Module):
    """Frozen sentence encoder for short task instructions."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        freeze: bool = True,
        max_length: int = 64,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.output_dim = int(self.encoder.config.hidden_size)
        self.freeze = freeze
        if freeze:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False

    @staticmethod
    def mean_pooling(
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        expanded_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * expanded_mask, dim=1)
        counts = torch.clamp(expanded_mask.sum(dim=1), min=1e-6)
        return summed / counts

    def forward(
        self,
        texts: Sequence[str],
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        if not texts:
            raise ValueError("texts must be a non-empty sequence")

        batch = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if device is None:
            device = next(self.encoder.parameters()).device
        batch = {key: value.to(device) for key, value in batch.items()}

        if self.freeze:
            with torch.no_grad():
                outputs = self.encoder(**batch)
        else:
            outputs = self.encoder(**batch)

        embeddings = self.mean_pooling(outputs.last_hidden_state, batch["attention_mask"])
        return torch.nn.functional.normalize(embeddings, p=2, dim=-1)

