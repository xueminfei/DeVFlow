"""Frozen molecular features in canonical two-drug slots (no learned drug IDs)."""
from pathlib import Path
import torch


def canonical_smiles(value):
    from rdkit import Chem
    molecule = Chem.MolFromSmiles(str(value))
    if molecule is None:
        raise ValueError(f"invalid SMILES: {value!r}")
    return Chem.MolToSmiles(molecule)


DRUG_DIM = 1024
CONCAT_DRUG_DIM = 2 * DRUG_DIM
MODEL_FAMILY = "map_original_se_decoder_joint_directconcat_otfm_v1"
CONDITION_POOL = "canonical_sorted_direct_concat_frozen_MAPKG_2x1024"


def canonical_drug_names(names):
    names = tuple(names)
    if len(names) not in (1, 2) or any(not isinstance(n, str) or not n for n in names):
        raise ValueError("Need one or two nonempty canonical drug names")
    if len(set(names)) != len(names):
        raise ValueError("Duplicate drugs are not valid ComboSciPlex conditions")
    return tuple(sorted(names))


def sample_conditions(sample, dataset, smiles_to_index, embedding_bank, device):
    """[e_A, e_B], canonical lexical order; monotherapy [e_A, zeros]."""
    if embedding_bank.ndim != 2 or embedding_bank.shape[1] != DRUG_DIM:
        raise ValueError("Expected frozen MAPKG bank [drugs,1024]")
    count = sample["control_state"].shape[0]
    names = canonical_drug_names(sample["drug_names"])
    tokens = torch.zeros(count, 2, DRUG_DIM, device=device, dtype=embedding_bank.dtype)
    for slot, name in enumerate(names):
        key = canonical_smiles(dataset.bundle["drug_smiles"][name])
        value = embedding_bank[smiles_to_index[key]].detach()
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite frozen drug vector: {name}")
        tokens[:, slot] = value
    drug = tokens.reshape(count, CONCAT_DRUG_DIM)
    dose = torch.zeros(count, device=device, dtype=drug.dtype)
    line = torch.zeros(count, 3, device=device, dtype=drug.dtype)
    line[:, 0] = 1
    return drug, dose, line


def load_drug_bank(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    names, bank = payload["canonical_smiles"], payload["embeddings"]
    if len(names) != len(set(names)) or bank.shape != (len(names), DRUG_DIM):
        raise ValueError("Drug bank identity/shape mismatch")
    if not torch.isfinite(bank).all():
        raise ValueError("Drug bank contains nonfinite features")
    return {name: index for index, name in enumerate(names)}, bank.detach().to(device)
