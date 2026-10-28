#!/usr/bin/env python3
"""G: materialize deterministic SE gene-token/count arrays for MAP training."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

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
    parser = argparse.ArgumentParser(description="Extract SciPlex3 SE inputs as .npy memmaps.")
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--se-config", required=True)
    parser.add_argument("--gene-embeddings", required=True)
    parser.add_argument("--cell-lines", nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--expr-dtype", choices=["float32", "float16"], default="float16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-cells", type=int, default=None, help="For smoke tests only")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cell_lines = resolve_cell_lines(args.cell_lines)
    cfg = OmegaConf.load(args.se_config)
    pad_length = int(cfg.dataset.pad_length)
    tokenizer, token_meta = create_tokenizer(
        args.manifest_dir,
        args.gene_embeddings,
        pad_length=pad_length,
        seed=args.seed,
    )
    matrix = DiskCSR(args.manifest_dir)
    cells = selected_cells(args.manifest_dir, cell_lines)
    out_root = Path(args.out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    token_path = out_root / "tokenization.json"
    if token_path.exists() and not args.overwrite:
        if load_json(token_path) != token_meta:
            raise ValueError("Tokenization settings changed; use --overwrite to restart stage G")
    else:
        dump_json(token_meta, token_path)

    for cell_line in cell_lines:
        line_cells = cells[cells["cell_line"] == cell_line].reset_index(drop=True)
        global_indices = line_cells["global_index"].to_numpy(dtype=np.int64)
        total = len(global_indices)
        stop = total if args.limit_cells is None else min(total, args.limit_cells)
        line_dir = out_root / cell_line
        line_dir.mkdir(parents=True, exist_ok=True)
        gene_path = line_dir / "se_gene_ids.npy"
        expr_path = line_dir / "se_expr.npy"
        shape_path = line_dir / "se_shape.json"
        progress_path = line_dir / "progress.json"
        if args.overwrite:
            for path in (gene_path, expr_path, shape_path, progress_path):
                if path.exists():
                    path.unlink()
        progress = load_json(progress_path) if progress_path.exists() else {"next_row": 0}
        assert isinstance(progress, dict)
        start = int(progress.get("next_row", 0))
        gene_output = open_npy_for_resume(
            gene_path, shape=(total, pad_length), dtype=np.int32, overwrite=False
        )
        expr_output = open_npy_for_resume(
            expr_path, shape=(total, pad_length), dtype=args.expr_dtype, overwrite=False
        )
        batches = range(start, stop, args.batch_size)
        for batch_start, gene_ids, expressions in tqdm(
            iter_token_batches(
                matrix,
                global_indices,
                tokenizer,
                batch_size=args.batch_size,
                start=start,
                stop=stop,
            ),
            total=len(batches),
            desc=f"G {cell_line}",
            unit="batch",
        ):
            batch_stop = batch_start + len(gene_ids)
            gene_output[batch_start:batch_stop] = gene_ids
            expr_output[batch_start:batch_stop] = expressions.astype(args.expr_dtype, copy=False)
            gene_output.flush()
            expr_output.flush()
            dump_json(
                {"next_row": batch_stop, "total_rows": total, "complete": batch_stop == total},
                progress_path,
            )

        complete = stop == total
        dump_json(
            {
                "N": total,
                "pad_length": pad_length,
                "gene_dtype": "int32",
                "expression_dtype": args.expr_dtype,
                "complete": complete,
            },
            shape_path,
        )
        print(f"G {cell_line}: SE inputs written under {line_dir}")
        if not complete:
            print(f"G {cell_line}: partial smoke-test output ({stop:,}/{total:,} rows)")

    print(f"G complete: use --se_inputs_base_dir {out_root}")


if __name__ == "__main__":
    main()
