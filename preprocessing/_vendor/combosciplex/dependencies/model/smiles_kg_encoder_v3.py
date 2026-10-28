import torch
import torch.nn as nn
from rdkit import Chem

from model.mega_molbart.STMencoder_ddp import MolSTM_Extractor


class ResidualProjector(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = (input_dim + output_dim) // 2
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )
        self.residual_proj = nn.Linear(input_dim, output_dim)
        self.gate = nn.Sequential(nn.Linear(input_dim, output_dim), nn.Sigmoid())

    def forward(self, x):
        transformed = self.mlp(x)
        residual = self.residual_proj(x)
        gate_value = self.gate(x)
        return gate_value * transformed + (1 - gate_value) * residual


class KGSmilesEncoder_v3(nn.Module):
    def __init__(self, mapkg_ckpt, vocab_path, d_smiles=256, d_model=1024):
        super().__init__()
        checkpoint = torch.load(mapkg_ckpt, map_location="cpu")
        state = checkpoint.get("model_state_dict", checkpoint)
        state = {
            (key[len("module."):] if key.startswith("module.") else key): value
            for key, value in state.items()
        }
        # Standalone MAP-KG checkpoints use short names; full MAP checkpoints
        # store this encoder below pert_model.kg_smiles_encoder.
        full_prefix = "pert_model.kg_smiles_encoder."
        if not any(key.startswith("smiles_encoder.") for key in state):
            state = {
                (key[len(full_prefix):] if key.startswith(full_prefix) else key): value
                for key, value in state.items()
            }
        molecule_state = {
            key[len("smiles_encoder._model."):]: value
            for key, value in state.items()
            if key.startswith("smiles_encoder._model.")
        }
        if not molecule_state:
            raise ValueError(f"No smiles_encoder._model weights found in {mapkg_ckpt}")
        self.smiles_encoder = MolSTM_Extractor(
            vocab_path=vocab_path,
            state_dict=molecule_state,
            pretrained=True,
        )
        self.smiles_projector = ResidualProjector(
            input_dim=d_smiles,
            output_dim=d_model,
            hidden_dim=512,
        )
        sub_state = {
            key: value
            for key, value in state.items()
            if key.startswith("smiles_encoder.") or key.startswith("smiles_projector.")
        }
        missing, unexpected = self.load_state_dict(sub_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"MAP-KG checkpoint mismatch: missing={missing}, unexpected={unexpected}"
            )
        del checkpoint, state, molecule_state, sub_state

    def forward(self, smiles_list):
        normalized = []
        for value in smiles_list:
            molecule = Chem.MolFromSmiles(value)
            normalized.append(Chem.MolToSmiles(molecule) if molecule is not None else value)
        if "xxx" in normalized:
            raise ValueError("invalid placeholder SMILES: xxx")
        return self.smiles_projector(self.smiles_encoder(normalized))
