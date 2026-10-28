#!/usr/bin/env python3
"""Reproducible frozen epoch-4 SE600M states; prepare-tokens, encode, merge.

Uses the repository's original SE architecture with tensor-preserving official
weights. Tokenization exactly follows SciPlexTokenizer (seed 42): merge symbols,
log1p(raw counts), divide by the sum over all mapped genes, multiply by 100,
retain the top 2047 mapped genes, and reserve the first position for CLS.
No HVG selection, fitting, or test labels enter these frozen states.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
MAP_ROOT = Path(__file__).resolve().parents[1] / 'state_encoder'
OFFICIAL_WEIGHT_SHA = '9d0cc6fc5aa89ed27c6c2ae336aa304ada4357518e40ad95351388e671d452e1'
LOSSLESS_CHECKPOINT_SHA = 'f462872681974de8e03dd22a3d4dd5199495622a54b2199d1a34cd245589ff20'
OFFICIAL_ESM_SHA = 'a210e1cc7901513999b2bca3836ba9e2f203cd008be4e9a9d6412a2267de9748'


def sha_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_hash(values):
    return hashlib.sha256('\n'.join(map(str, values)).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def cell_order(bundle):
    with (bundle / 'obs.csv').open(newline='') as f:
        values = [row['cell_id'] for row in csv.DictReader(f)]
    digest = ordered_hash(values)
    manifest = read_json(bundle / 'manifest.json')
    if digest != manifest['cell_order_sha256']:
        raise ValueError('obs.csv cell ordering does not match manifest')
    return values, digest


class Tokenizer:
    """Deterministic, symbol-aggregating original SciPlex tokenizer."""
    def __init__(self, gene_to_token, num_tokens, seed=42, pad_length=2048):
        self.gene_to_token = np.asarray(gene_to_token, dtype=np.int32)
        self.pad_length = pad_length
        rng = np.random.default_rng(seed)
        self.fill_order = rng.permutation(num_tokens).astype(np.int32)
        self.tie_priority = rng.permutation(num_tokens).astype(np.int32)
        self.expressed_mask = np.zeros(num_tokens, dtype=bool)

    def encode(self, indices, raw_counts):
        tokens = self.gene_to_token[np.asarray(indices, dtype=np.int64)]
        counts = np.asarray(raw_counts, dtype=np.float64)
        valid = (tokens >= 0) & (counts > 0)
        tokens, counts = tokens[valid], counts[valid]
        gene_ids = np.zeros(self.pad_length, dtype=np.int32)
        expressions = np.zeros(self.pad_length, dtype=np.float32)
        if len(tokens):
            order = np.argsort(tokens, kind='stable')
            sorted_tokens, sorted_counts = tokens[order], counts[order]
            starts = np.flatnonzero(np.r_[True, sorted_tokens[1:] != sorted_tokens[:-1]])
            tokens, counts = sorted_tokens[starts], np.add.reduceat(sorted_counts, starts)
            transformed = np.log1p(counts)
            weights = transformed / transformed.sum()
            keep = np.lexsort((self.tie_priority[tokens], -transformed))[:self.pad_length - 1]
            selected = tokens[keep]
            gene_ids[1:len(selected) + 1] = selected
            expressions[1:len(selected) + 1] = (100 * weights[keep]).astype(np.float32)
        else:
            selected = np.empty(0, dtype=np.int32)
        remaining = self.pad_length - 1 - len(selected)
        if remaining:
            self.expressed_mask[selected] = True
            fillers = self.fill_order[~self.expressed_mask[self.fill_order]][:remaining]
            gene_ids[len(selected) + 1:] = fillers
            self.expressed_mask[selected] = False
        return gene_ids, expressions


def prepare_tokens(args):
    import h5py
    from scipy import sparse
    import torch
    bundle = Path(args.bundle_dir).resolve()
    cache = bundle / 'se_cache'
    cache.mkdir(parents=True, exist_ok=True)
    cells, digest = cell_order(bundle)
    source_sha = sha_file(args.source_h5ad)
    if source_sha != read_json(bundle / 'manifest.json')['source_sha256']:
        raise ValueError('Source h5ad SHA256 differs from prepared bundle')
    esm_sha = sha_file(args.gene_embeddings)
    if esm_sha != OFFICIAL_ESM_SHA:
        raise ValueError('Expected the original official ESM2 dictionary')
    protein = torch.load(args.gene_embeddings, map_location='cpu', weights_only=False)
    symbols = list(protein.keys())
    del protein
    with h5py.File(args.source_h5ad, 'r') as f:
        source_cells = f['obs'][f['obs'].attrs['_index']].asstr()[...].tolist()
        if source_cells != cells:
            raise ValueError('Source h5ad and bundle cell IDs/order differ')
        genes = f['var'][f['var'].attrs['_index']].asstr()[...].tolist()
        group = f['layers/counts']
        shape = tuple(int(v) for v in group.attrs['shape'])
        cls = sparse.csc_matrix if group.attrs['encoding-type'] == 'csc_matrix' else sparse.csr_matrix
        matrix = cls((group['data'][...], group['indices'][...], group['indptr'][...]), shape=shape).tocsr()
    token_lookup = {symbol: idx for idx, symbol in enumerate(symbols)}
    mapping = np.array([token_lookup.get(symbol, -1) for symbol in genes], dtype=np.int32)
    meta = dict(cell_order_sha256=digest, source_sha256=source_sha,
                protein_embeddings_sha256=esm_sha, protein_gene_order_sha256=ordered_hash(symbols),
                seed=args.seed, pad_length=2048, n_cells=len(cells), n_source_genes=len(genes),
                n_mapped_source_genes=int((mapping >= 0).sum()), n_tokens=len(symbols),
                transform='100 * log1p(raw_symbol_aggregated_counts) / sum(log1p(raw_symbol_aggregated_counts)) over all mapped genes; top2047 plus CLS',
                counts_source='layers/counts', complete=False)
    progress_path = cache / 'tokenization.json'
    if progress_path.exists():
        old = read_json(progress_path)
        for key in ('cell_order_sha256', 'source_sha256', 'protein_embeddings_sha256', 'seed'):
            if old[key] != meta[key]:
                raise ValueError('Tokenization provenance changed: ' + key)
        if old['complete']:
            print(json.dumps(old, indent=2), flush=True)
            return
    tokenizer = Tokenizer(mapping, len(symbols), args.seed)
    ids = np.lib.format.open_memmap(cache / 'token_ids.npy', mode='w+', dtype='int32', shape=(len(cells), 2048))
    values = np.lib.format.open_memmap(cache / 'token_counts.npy', mode='w+', dtype='float32', shape=(len(cells), 2048))
    start_time = time.time()
    for i in range(len(cells)):
        a, b = matrix.indptr[i:i + 2]
        ids[i], values[i] = tokenizer.encode(matrix.indices[a:b], matrix.data[a:b])
        if (i + 1) % 10000 == 0:
            print(f'Tokenized {i + 1}/{len(cells)} cells in {time.time() - start_time:.1f}s', flush=True)
    ids.flush(); values.flush()
    if not np.isfinite(values).all():
        raise ValueError('Non-finite token expression')
    meta.update(complete=True, seconds=time.time() - start_time,
                token_ids_sha256=sha_file(cache / 'token_ids.npy'),
                token_counts_sha256=sha_file(cache / 'token_counts.npy'))
    write_json(progress_path, meta)
    print(json.dumps(meta, indent=2), flush=True)


def load_model(checkpoint, config, device):
    import torch
    from omegaconf import OmegaConf
    sys.path.insert(0, str(MAP_ROOT))
    from model.se import StateEmbeddingModel
    loaded = torch.load(checkpoint, map_location='cpu', weights_only=False)
    kwargs = dict(loaded['hyper_parameters'])
    cfg = OmegaConf.create(loaded['cfg_yaml'])
    if config:
        requested_cfg = OmegaConf.load(config)
        for key in ('emsize', 'nhead', 'd_hid', 'nlayers', 'output_dim', 'counts', 'dataset_correction', 'use_flash_attention'):
            if requested_cfg.model.get(key) != cfg.model.get(key):
                raise ValueError('SE architecture config differs from checkpoint: ' + key)
    kwargs.update(cfg=cfg, dropout=0.0, compiled=False)
    model = StateEmbeddingModel(**kwargs)
    state = dict(loaded['state_dict'])
    model.pe_embedding = torch.nn.Embedding.from_pretrained(state['pe_embedding.weight'], freeze=True)
    for suffix in ('0.weight', '0.bias', '1.weight', '1.bias'):
        source, alias = 'encoder.' + suffix, 'gene_embedding_layer.' + suffix
        if alias not in state:
            state[alias] = state[source]
        elif not torch.equal(state[alias], state[source]):
            raise ValueError('Inconsistent shared encoder aliases')
    state.setdefault('_bin_indices_cached', torch.arange(10))
    model.load_state_dict(state, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval().to(device)
    return model


def infer(model, gene_ids, expressions, device, dtype):
    import torch
    from torch.nn import functional as F
    # Same operations as SE.infer, with an explicit precision context; fp16 matches
    # the legacy inner torch.cuda.amp.autocast() used by SE.infer.
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == 'cuda'):
        ids = torch.from_numpy(np.array(gene_ids, copy=True)).long().to(device)
        counts = torch.from_numpy(np.array(expressions, copy=True)).float().to(device)
        src = F.normalize(model.pe_embedding(ids), dim=2)
        src = torch.cat((model.cls_token.expand(len(ids), 1, -1), src[:, 1:, :]), dim=1)
        if model.dataset_token is not None:
            src = torch.cat((src, model.dataset_token.expand(len(ids), 1, -1)), dim=1)
        _, embeddings, _ = model.forward(src=src, counts=counts)
        return embeddings.float().cpu().numpy()


def encode(args):
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(42)
    bundle = Path(args.bundle_dir).resolve()
    cache = bundle / 'se_cache'
    cells, digest = cell_order(bundle)
    token_meta = read_json(cache / 'tokenization.json')
    if not token_meta['complete'] or token_meta['cell_order_sha256'] != digest:
        raise ValueError('Prepare complete aligned tokens first')
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError('Invalid shard index')
    first = len(cells) * args.shard_index // args.num_shards
    last = len(cells) * (args.shard_index + 1) // args.num_shards
    stop = last if args.limit_cells is None else min(first + args.limit_cells, last)
    shard_name = f'shard_{args.shard_index:02d}_of_{args.num_shards:02d}'
    progress_path = cache / (shard_name + '.json')
    output_path = cache / (shard_name + '.npy')
    checkpoint_sha = sha_file(args.checkpoint)
    if checkpoint_sha != LOSSLESS_CHECKPOINT_SHA:
        raise ValueError('Expected the audited lossless official epoch4 checkpoint')
    protocol = dict(cell_order_sha256=digest, checkpoint_sha256=checkpoint_sha,
                    official_safetensors_sha256=OFFICIAL_WEIGHT_SHA,
                    token_ids_sha256=token_meta['token_ids_sha256'], token_counts_sha256=token_meta['token_counts_sha256'],
                    first_row=first, last_row=last, n_shards=args.num_shards, shard_index=args.shard_index,
                    shape=[last - first, 2048], dtype='float32', amp_dtype=args.amp_dtype,
                    dropout=0.0, model_code_sha256=sha_file(MAP_ROOT / 'model/se.py'),
                    attention_code_sha256=sha_file(MAP_ROOT / 'model/flash_transformer.py'))
    if progress_path.exists():
        old = read_json(progress_path)
        if any(old.get(k) != v for k, v in protocol.items()):
            raise ValueError('Cannot resume shard with different provenance')
        next_row = old['next_row']
        output = np.load(output_path, mmap_mode='r+')
        if list(output.shape) != protocol['shape']:
            raise ValueError('Shard shape changed')
    else:
        next_row = first
        output = np.lib.format.open_memmap(output_path, mode='w+', dtype='float32', shape=(last - first, 2048))
        output[:] = np.nan
        output.flush()
    if next_row >= stop:
        print(f'{shard_name} already at {next_row}/{last}', flush=True)
        return
    device = torch.device(args.device)
    if device.type == 'cuda':
        free, total = torch.cuda.mem_get_info(device)
        if free < 8 * 1024**3:
            raise RuntimeError(f'Need at least 8 GiB free CUDA memory, available {free / 1024**3:.2f} GiB')
        print(f'{device}: free {free / 1024**3:.2f}/{total / 1024**3:.2f} GiB', flush=True)
    model = load_model(args.checkpoint, args.se_config, device)
    ids = np.load(cache / 'token_ids.npy', mmap_mode='r')
    counts = np.load(cache / 'token_counts.npy', mmap_mode='r')
    dtype = torch.bfloat16 if args.amp_dtype == 'bf16' else torch.float16
    start_time, initial_row = time.time(), next_row
    protocol.update(batch_size=args.batch_size, device=str(device))
    for start in range(next_row, stop, args.batch_size):
        end = min(start + args.batch_size, stop)
        embeddings = infer(model, ids[start:end], counts[start:end], device, dtype)
        if embeddings.shape != (end - start, 2048) or not np.isfinite(embeddings).all():
            raise ValueError(f'Invalid embeddings at {start}:{end}')
        output[start - first:end - first] = embeddings
        output.flush()
        elapsed = time.time() - start_time
        progress = dict(protocol, next_row=end, complete=end == last,
                        seconds_this_session=elapsed, cells_per_second=(end - initial_row) / max(elapsed, 1e-6),
                        max_cuda_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == 'cuda' else 0,
                        max_cuda_reserved_gib=torch.cuda.max_memory_reserved(device) / 1024**3 if device.type == 'cuda' else 0)
        write_json(progress_path, progress)
        if end == stop or (end - initial_row) % (args.batch_size * 25) == 0:
            print(json.dumps(progress), flush=True)


def merge(args):
    import importlib.metadata
    bundle = Path(args.bundle_dir).resolve()
    cache = bundle / 'se_cache'
    cells, digest = cell_order(bundle)
    output_path = bundle / 'se_embeddings.npy'
    partial = bundle / 'se_embeddings.partial.npy'
    output = np.lib.format.open_memmap(partial, mode='w+', dtype='float32', shape=(len(cells), 2048))
    records = []
    for shard in range(args.num_shards):
        name = f'shard_{shard:02d}_of_{args.num_shards:02d}'
        record = read_json(cache / (name + '.json'))
        first, last = len(cells) * shard // args.num_shards, len(cells) * (shard + 1) // args.num_shards
        if not record['complete'] or record['next_row'] != last or record['first_row'] != first or record['last_row'] != last or record['cell_order_sha256'] != digest:
            raise ValueError('Incomplete/misaligned shard: ' + name)
        values = np.load(cache / (name + '.npy'), mmap_mode='r')
        if values.shape != (last - first, 2048) or not np.isfinite(values).all():
            raise ValueError('Non-finite/wrong-shape shard: ' + name)
        if records:
            for key in ('checkpoint_sha256', 'token_ids_sha256', 'token_counts_sha256', 'amp_dtype', 'model_code_sha256', 'attention_code_sha256'):
                if records[0][key] != record[key]:
                    raise ValueError('Different encoder protocol across shards: ' + key)
        output[first:last] = values
        records.append(record)
    output.flush()
    norms = np.linalg.norm(output, axis=1)
    if not np.isfinite(norms).all() or np.max(np.abs(norms - 1)) > 0.02:
        raise ValueError('SE states should be finite unit-normalized vectors')
    del output
    os.replace(partial, output_path)
    audit = dict(shape=[len(cells), 2048], dtype='float32', finite=True,
                 cell_order_sha256=digest, state_sha256=sha_file(output_path),
                 checkpoint_sha256=records[0]['checkpoint_sha256'], official_safetensors_sha256=OFFICIAL_WEIGHT_SHA,
                 norm_min=float(norms.min()), norm_max=float(norms.max()),
                 software=dict(python=sys.version, **{name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'scipy', 'h5py', 'lightning', 'omegaconf')}),
                 tokenization=read_json(cache / 'tokenization.json'), shards=records)
    write_json(bundle / 'se_cache_audit.json', audit)
    manifest = read_json(bundle / 'manifest.json')
    manifest.update(state_embeddings_present=True, se_cell_order_sha256=digest,
                    state_embeddings_sha256=audit['state_sha256'], missing_requirements=[],
                    state_encoder_checkpoint_sha256=audit['checkpoint_sha256'],
                    state_cache_audit='se_cache_audit.json')
    write_json(bundle / 'manifest.json', manifest)
    print(json.dumps({k: v for k, v in audit.items() if k not in ('tokenization', 'shards')}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare-tokens')
    prep.add_argument('--source-h5ad', required=True)
    prep.add_argument('--gene-embeddings', required=True)
    prep.add_argument('--seed', type=int, default=42)
    enc = sub.add_parser('encode')
    enc.add_argument('--checkpoint', required=True)
    enc.add_argument('--se-config')
    enc.add_argument('--device', default='cuda:0')
    enc.add_argument('--batch-size', type=int, default=4)
    enc.add_argument('--num-shards', type=int, default=8)
    enc.add_argument('--shard-index', type=int, required=True)
    enc.add_argument('--limit-cells', type=int)
    enc.add_argument('--amp-dtype', choices=['bf16', 'fp16'], default='fp16')
    final = sub.add_parser('merge')
    final.add_argument('--num-shards', type=int, default=8)
    for action in (prep, enc, final):
        action.add_argument('--bundle-dir', required=True)
    args = parser.parse_args()
    {'prepare-tokens': prepare_tokens, 'encode': encode, 'merge': merge}[args.command](args)


if __name__ == '__main__':
    main()
