"""Frozen MAP-KG drug encoding for flow-matching conditions."""

from collections import OrderedDict

import torch
from rdkit import Chem

from model.smiles_kg_encoder_v3 import KGSmilesEncoder_v3


def canonical_smiles(value):
    molecule = Chem.MolFromSmiles(str(value))
    if molecule is None:
        raise ValueError(f"invalid SMILES: {value!r}")
    return Chem.MolToSmiles(molecule)


def encode_mapkg_smiles(smiles, checkpoint, vocabulary, device, batch_size=64):
    """Encode every unique molecule once and return canonical SMILES plus CPU embeddings."""
    canonical = list(OrderedDict.fromkeys(canonical_smiles(value) for value in smiles))
    if not canonical:
        raise ValueError("at least one SMILES is required")
    encoder = KGSmilesEncoder_v3(
        mapkg_ckpt=checkpoint,
        vocab_path=vocabulary,
        d_smiles=256,
        d_model=1024,
    ).to(device)
    encoder.requires_grad_(False)
    encoder.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(canonical), batch_size):
            encoded = encoder(canonical[start:start + batch_size])
            outputs.append(encoded.float().cpu())
    embeddings = torch.cat(outputs, dim=0)
    if embeddings.shape != (len(canonical), 1024) or not torch.isfinite(embeddings).all():
        raise ValueError(f"invalid MAP-KG output: {tuple(embeddings.shape)}")
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return canonical, embeddings

