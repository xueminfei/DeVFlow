#!/usr/bin/env python3
"""D: extract frozen SE600M cell embeddings for every selected SciPlex3 cell."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from data.utils import get_embedding_cfg
from model.cell_emb import CellEmbeddingExtractor
from sciplex_utils import (
    DiskCSR,
    create_tokenizer,
    dump_json,
    iter_token_batches,
    load_json,
    open_npy_for_resume,
    resolve_cell_lines,
    selected_cells,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract SE600M supervision embeddings for SciPlex3.")
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--se-config", required=True)
    parser.add_argument("--se-ckpt", required=True)
    parser.add_argument("--gene-embeddings", required=True)
    parser.add_argument("--cell-lines", nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--amp-dtype", choices=["none", "bf16", "fp16"], default="bf16")
    parser.add_argument("--output-dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-cells", type=int, default=None, help="For smoke tests only")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _autocast(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "none":
        return nullcontext()
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    cell_lines = resolve_cell_lines(args.cell_lines)
    device = _resolve_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    cfg = OmegaConf.load(args.se_config)
    get_embedding_cfg(cfg).all_embeddings = str(Path(args.gene_embeddings).resolve())
    pad_length = int(cfg.dataset.pad_length)
    tokenizer, token_meta = create_tokenizer(
        args.manifest_dir, args.gene_embeddings, pad_length=pad_length, seed=args.seed
    )
    matrix = DiskCSR(args.manifest_dir)
    cells = selected_cells(args.manifest_dir, cell_lines)
    extractor = CellEmbeddingExtractor(
        config=args.se_config,
        se_model_path=args.se_ckpt,
        gene_embeddings=args.gene_embeddings,
        device=device,
    )
    model = extractor.se
    out_root = Path(args.out_dir).expanduser().resolve()

    for cell_line in cell_lines:
        line_cells = cells[cells["cell_line"] == cell_line].reset_index(drop=True)
        global_indices = line_cells["global_index"].to_numpy(dtype=np.int64)
        total = len(global_indices)
        stop = total if args.limit_cells is None else min(total, args.limit_cells)
        line_dir = out_root / cell_line
        line_dir.mkdir(parents=True, exist_ok=True)
        embedding_path = line_dir / f"{cell_line}_embeddings.npy"
        progress_path = line_dir / "progress.json"
        token_path = line_dir / "tokenization.json"
        if args.overwrite:
            for path in (embedding_path, progress_path, token_path):
                if path.exists():
                    path.unlink()
        progress = load_json(progress_path) if progress_path.exists() else {"next_row": 0}
        assert isinstance(progress, dict)
        start = int(progress.get("next_row", 0))
        if start > stop:
            raise ValueError(f"Resume row {start} is beyond requested stop {stop} for {cell_line}")
        output = open_npy_for_resume(
            embedding_path,
            shape=(total, 2048),
            dtype=args.output_dtype,
            overwrite=False,
        )
        if token_path.exists():
            existing_token_meta = load_json(token_path)
            if existing_token_meta != token_meta:
                raise ValueError(
                    f"Tokenization settings changed while resuming {cell_line}; "
                    "use --overwrite to restart"
                )
        else:
            dump_json(token_meta, token_path)

        batches = range(start, stop, args.batch_size)
        progress_bar = tqdm(total=len(batches), desc=f"D {cell_line}", unit="batch")
        for batch_start, gene_ids, expressions in iter_token_batches(
            matrix,
            global_indices,
            tokenizer,
            batch_size=args.batch_size,
            start=start,
            stop=stop,
        ):
            gene_tensor = torch.from_numpy(gene_ids).long()
            expression_tensor = torch.from_numpy(expressions).float()
            with torch.inference_mode(), _autocast(device, args.amp_dtype):
                embedding = model.infer((gene_tensor, expression_tensor))
            values = embedding.detach().float().cpu().numpy().astype(args.output_dtype, copy=False)
            batch_stop = batch_start + len(values)
            output[batch_start:batch_stop] = values
            output.flush()
            dump_json(
                {
                    "next_row": batch_stop,
                    "total_rows": total,
                    "complete": batch_stop == total,
                    "embedding_dim": 2048,
                    "dtype": args.output_dtype,
                },
                progress_path,
            )
            progress_bar.update(1)
        progress_bar.close()
        print(f"D {cell_line}: wrote {stop - start:,} embeddings to {embedding_path}")
        if stop < total:
            print(f"D {cell_line}: partial smoke-test output ({stop:,}/{total:,} rows)")


if __name__ == "__main__":
    main()
