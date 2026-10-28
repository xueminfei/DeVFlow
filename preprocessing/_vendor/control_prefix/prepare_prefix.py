#!/usr/bin/env python3
"""One fixed native SE prefix realization, controls only, shared by both arms."""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(name, '1')
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

MAP = Path(__file__).resolve().parents[1] / 'state_encoder'
sys.path.insert(0, str(MAP))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def write(path, value):
    tmp = Path(str(path)+'.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--seed', type=int, default=20261021)
    p.add_argument('--processed-dir', type=Path, required=True)
    p.add_argument('--tokens-dir', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--se-config', type=Path, required=True)
    p.add_argument('--expected-world-size', type=int, default=8,
                   help='Historical cache used 8 ranks; changing this changes its stochastic realization.')
    a = p.parse_args()
    import numpy as np
    import torch
    from torch import nn
    import torch.distributed as dist
    from omegaconf import OmegaConf
    from model.se import StateEmbeddingModel
    from se_control_lora import from_se_model, prefix_from_tokens
    DATA = a.processed_dir.expanduser().resolve()
    TOKENS = a.tokens_dir.expanduser().resolve()
    CKPT = a.checkpoint.expanduser().resolve()
    CFG = a.se_config.expanduser().resolve()
    rank = int(os.environ.get('RANK', '0'))
    local = int(os.environ.get('LOCAL_RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if world != a.expected_world_size:
        raise ValueError(f'Expected {a.expected_world_size} ranks; launch with torchrun or explicitly change --expected-world-size.')
    torch.set_num_threads(1)
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    if world > 1:
        dist.init_process_group('nccl')
    out = a.output.resolve()
    cache = out/'control_prefix_cache'
    shards = cache/'shards'
    shards.mkdir(parents=True, exist_ok=True)
    if (cache/'manifest.json').exists():
        raise FileExistsError('A completed immutable prefix cache already exists')
    seed = a.seed+rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    start = time.time()
    if rank == 0:
        write(out/'prefix_status.json', dict(status='extracting', world_size=world, seed=a.seed,
              batch_size=a.batch_size, started_at=start))
    cfg = OmegaConf.load(CFG)
    ckpt = torch.load(str(CKPT), map_location='cpu', mmap=True, weights_only=False)
    state = {k[3:]: v for k, v in ckpt['model_state_dict'].items() if k.startswith('se.')}
    with torch.device('meta'):
        se = StateEmbeddingModel(token_dim=5120, d_model=2048, nhead=16, d_hid=2048,
             nlayers=16, output_dim=2048, dropout=.1, warmup_steps=0, compiled=False,
             max_lr=cfg.optimizer.max_lr, emb_size=5120, cfg=cfg, collater=None)
        se.pe_embedding = nn.Embedding(*state['pe_embedding.weight'].shape)
    se.load_state_dict(state, strict=True, assign=True)
    del state, ckpt
    se = se.to(device).eval().requires_grad_(False)
    with torch.random.fork_rng(devices=[local]):
        torch.manual_seed(42)
        tail = from_se_model(se, rank=8, alpha=8, allow_stochastic_attention=True).eval()
    if rank == 0:
        tail.cpu().save_tail(out/'se_tail_init.pt', metadata=dict(
            source=str(CKPT), source_se_prefix='se.', native_attention_dropout=.1,
            control_only=True, cache_seed=a.seed, source_code_sha256=sha(Path(__file__))))
        tail.to(device)
    reports = {}
    for line in ('A549', 'MCF7', 'K562'):
        with (DATA/line/f'{line}_meta.pkl').open('rb') as f:
            meta = pickle.load(f)
        ids = np.asarray(meta['ds_level_index'], dtype=np.int64)
        is_control = np.asarray(meta['is_control'], dtype=bool)
        allowed = np.asarray(json.loads((DATA/line/'control_indices.json').read_text()), dtype=np.int64)
        all_rows = np.flatnonzero(np.isin(ids, allowed)).astype(np.int64)
        assert len(all_rows) > 0 and is_control[all_rows].all()
        assert set(ids[all_rows]) == set(allowed)
        positions = np.arange(len(all_rows), dtype=np.int64)[rank::world]
        rows = all_rows[positions]
        genes = np.load(TOKENS/line/'se_gene_ids.npy', mmap_mode='r')
        expr = np.load(TOKENS/line/'se_expr.npy', mmap_mode='r')
        assert genes.shape == expr.shape == (len(ids), 2048)
        values = []
        parity = None
        for j in range(0, len(rows), a.batch_size):
            chosen = rows[j:j+a.batch_size]
            g = torch.from_numpy(np.array(genes[chosen], dtype=np.int64)).to(device)
            e = torch.from_numpy(np.array(expr[chosen], dtype=np.float32)).to(device)
            rng_before = torch.cuda.get_rng_state(device)
            with torch.inference_mode():
                pre = prefix_from_tokens(se, g, e, allow_stochastic_attention=True)
            assert pre.shape == (len(chosen), 2048) and torch.isfinite(pre).all()
            values.append(pre.float().cpu().numpy())
            if j == 0:
                rng_after = torch.cuda.get_rng_state(device)
                torch.cuda.set_rng_state(rng_before, device)
                with torch.inference_mode():
                    full = se.infer((g, e)).float()
                    with torch.cuda.amp.autocast():
                        replay = tail.base_forward(pre).float()
                torch.cuda.set_rng_state(rng_after, device)
                cosine = torch.nn.functional.cosine_similarity(full, replay, dim=-1)
                parity = dict(max_abs_error=float((full-replay).abs().max()),
                              cosine_min=float(cosine.min()), cosine_mean=float(cosine.mean()))
                assert parity['cosine_min'] > .99999, parity
            if j % (a.batch_size*50) == 0:
                print(f'prefix rank={rank} {line} {j+len(chosen)}/{len(rows)} elapsed={time.time()-start:.1f}s', flush=True)
        np.savez(shards/f'{line}_rank_{rank}.npz', positions=positions, rows=rows,
                 prefix=np.concatenate(values), global_ids=ids[rows])
        reports[line] = dict(cells=len(rows), same_realization_full_vs_tail=parity)
    write(shards/f'rank_{rank}_report.json', dict(rank=rank, seed=seed, lines=reports,
          gpu=torch.cuda.get_device_name(local), torch=torch.__version__, elapsed_seconds=time.time()-start))
    del se, tail
    torch.cuda.empty_cache()
    if world > 1:
        dist.barrier()
    if rank == 0:
        manifest = dict(status='complete', controls_only=True, cache_seed=a.seed, world_size=world,
             partition='sorted control rows, rank-strided within each line', batch_size=a.batch_size,
             native_attention_dropout=.1, frozen_prefix_realization=True,
             native_nn_dropout='eval disabled', extraction_precision='native infer inner FP16 autocast',
             online_tail_precision='parent BF16 autocast during training; FP32 evaluation',
             attention_behavior_unchanged=True, target_embeddings='original frozen processed cache',
             source_checkpoint=str(CKPT), source_checkpoint_sha256=sha(CKPT),
             encoder_source_sha256=sha(MAP/'model/se.py'), flash_source_sha256=sha(MAP/'model/flash_transformer.py'),
             producer_sha256=sha(__file__), tail_module_sha256=sha(Path(__file__).with_name('se_control_lora.py')),
             tail_checkpoint_sha256=sha(out/'se_tail_init.pt'), lines={},
             rank_reports=[json.loads((shards/f'rank_{r}_report.json').read_text()) for r in range(world)])
        for line in ('A549', 'MCF7', 'K562'):
            parts = [np.load(shards/f'{line}_rank_{r}.npz') for r in range(world)]
            positions = np.concatenate([s['positions'] for s in parts])
            order = np.argsort(positions)
            np.testing.assert_array_equal(positions[order], np.arange(len(positions)))
            rows = np.concatenate([s['rows'] for s in parts])[order]
            prefix = np.concatenate([s['prefix'] for s in parts])[order].astype(np.float32)
            reference = np.array(np.load(DATA/line/f'{line}_embeddings.npy', mmap_mode='r')[rows], dtype=np.float32)
            dest = cache/line
            dest.mkdir(exist_ok=True)
            np.save(dest/'rows.npy', rows)
            np.save(dest/'prefix.npy', prefix)
            np.save(dest/'reference_embeddings.npy', reference)
            manifest['lines'][line] = dict(cells=len(rows), prefix_shape=list(prefix.shape),
                 rows_sha256=sha(dest/'rows.npy'), prefix_sha256=sha(dest/'prefix.npy'),
                 reference_sha256=sha(dest/'reference_embeddings.npy'),
                 tokens_gene_sha256=sha(TOKENS/line/'se_gene_ids.npy'),
                 tokens_expression_sha256=sha(TOKENS/line/'se_expr.npy'))
            for part in parts:
                part.close()
        assert sum(s['cells'] for s in manifest['lines'].values()) == 12708
        manifest['elapsed_seconds'] = time.time()-start
        write(cache/'manifest.json', manifest)
        write(out/'prefix_status.json', dict(status='complete', cells=12708, elapsed_seconds=time.time()-start))
        print('PREFIX_COMPLETE', json.dumps(dict(cells=12708, elapsed_seconds=time.time()-start)), flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
