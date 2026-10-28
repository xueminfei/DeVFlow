from pathlib import Path
import runpy, sys
import numpy as np
import scanpy.preprocessing._highly_variable_genes as hv
import skmisc.loess as loess_module
from skmisc.loess import loess as _loess

class DeterministicJitterLoess:
    def __init__(self, x, y, *args, **kwargs):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        order = np.argsort(x, kind="mergesort")
        xs = x[order].copy()
        if len(xs) > 1:
            scale = max(1.0, float(np.max(np.abs(xs))))
            eps = 1e-8 * scale
            runs = np.r_[0, np.flatnonzero(np.diff(xs) == 0) + 1, len(xs)]
            for a, b in zip(runs[:-1], runs[1:]):
                if b - a > 1:
                    xs[a:b] += eps * np.arange(b - a, dtype=np.float64)
            x2 = np.empty_like(xs); x2[order] = xs
            x = x2
        self._model = _loess(x, y, *args, **kwargs)
    def __getattr__(self, name):
        return getattr(self._model, name)

loess_module.loess = DeterministicJitterLoess
print("PATCHED_LOESS", loess_module.loess, flush=True)
script = Path(__file__).resolve().parents[1] / "sciplex3/E_hvg_multi_celllines.py"
sys.path.insert(0, str(script.parent)); sys.argv = sys.argv
runpy.run_path(str(script), run_name="__main__")
