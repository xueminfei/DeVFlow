"""Loader for the frozen SE600M encoder used during preprocessing."""

from __future__ import annotations

from collections import OrderedDict

import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file
from torch import nn

from data.utils import get_embedding_cfg
from model.se import StateEmbeddingModel


class CellEmbeddingExtractor:
    def __init__(
        self,
        config: str,
        se_model_path: str,
        gene_embeddings: str | None = None,
        device: str | torch.device | None = None,
    ):
        self.cfg = OmegaConf.load(config)
        if gene_embeddings is not None:
            get_embedding_cfg(self.cfg).all_embeddings = str(gene_embeddings)
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.se = self._load_se(se_model_path)
        self.se.eval()
        for parameter in self.se.parameters():
            parameter.requires_grad_(False)
        self.se.to(self.device)
        print(f"Initialized frozen SE model on {self.device}")

    def _load_se(self, model_path: str) -> StateEmbeddingModel:
        model = StateEmbeddingModel(
            token_dim=5120,
            d_model=2048,
            nhead=16,
            d_hid=2048,
            nlayers=16,
            output_dim=2048,
            dropout=0.1,
            warmup_steps=0,
            compiled=False,
            max_lr=self.cfg.optimizer.max_lr,
            emb_size=5120,
            collater=None,
            cfg=self.cfg,
        )
        all_pe = torch.load(
            get_embedding_cfg(self.cfg).all_embeddings, map_location="cpu", weights_only=False
        )
        if isinstance(all_pe, dict):
            all_pe = torch.vstack(list(all_pe.values()))
        all_pe = all_pe.float()
        all_pe.requires_grad = False
        model.pe_embedding = nn.Embedding.from_pretrained(all_pe)

        state = load_file(model_path, device="cpu")
        wanted = set(model.state_dict().keys())
        new_state = OrderedDict(state)
        pairs = [
            ("encoder.0.weight", "gene_embedding_layer.0.weight"),
            ("encoder.0.bias", "gene_embedding_layer.0.bias"),
            ("encoder.1.weight", "gene_embedding_layer.1.weight"),
            ("encoder.1.bias", "gene_embedding_layer.1.bias"),
        ]
        for source, target in pairs:
            if source in state and target in wanted and target not in new_state:
                new_state[target] = state[source]
            if target in state and source in wanted and source not in new_state:
                new_state[source] = state[target]

        missing, unexpected = model.load_state_dict(new_state, strict=False)
        expected_missing = {"_bin_indices_cached"}
        critical_missing = [key for key in missing if key not in expected_missing]
        if critical_missing or unexpected:
            raise RuntimeError(
                f"SE checkpoint mismatch; missing={critical_missing}, unexpected={list(unexpected)}"
            )
        print(f"Loaded SE checkpoint from {model_path}")
        return model
