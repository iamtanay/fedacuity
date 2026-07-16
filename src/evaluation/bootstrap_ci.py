"""
FedAcuity -- Paired instance-level bootstrap CIs for held-out AUC.

Why this exists: the FL pipeline is fully deterministic under SEED=42 (each
client retrains with a fixed tree budget every round), so per-round AUCs are
constant after round 1. Testing significance ACROSS rounds (Mann-Whitney on
per-round AUC) is therefore pseudo-replication -- a test between two constants.
The correct uncertainty statement is at the TEST-INSTANCE level: a paired
bootstrap over the pooled held-out test set, resampling the same instances for
every strategy so the AUC difference accounts for error correlation.

Outputs -> results/tables/bootstrap_ci.json:
  per-strategy AUC with 95% CI (clustered_fl, fedavg, centralised)
  paired delta CFL - FedAvg: overall and per held-out facility

Usage:
    python -m src.evaluation.bootstrap_ci
"""

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from src.config import cfg
from src.data.loaders import load_held_out
from src.data.schema import FACILITY_CARE_TYPES
from src.fl.client import deserialize_xgb_model
from src.evaluation.eval_held_out import _run_fl_50rounds

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TABLES_DIR = Path(cfg["paths"]["results"]["tables"])
LOGS_DIR = Path(cfg["paths"]["results"]["logs"])
SEED = cfg["project"]["seed"]
XGB_CFG = cfg["fl"]["xgboost"]
N_BOOT = 2000


def _probs_global(client_results) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Weighted-ensemble probabilities of a flat client pool on each held-out facility."""
    held = load_held_out()
    total = sum(n for _, n in client_results.values())
    out = {}
    for fid, sp in held.items():
        X, y = sp["test"]
        p = np.zeros(len(X))
        for _, (b, n) in client_results.items():
            p += deserialize_xgb_model(b).predict_proba(X.values)[:, 1] * (n / total)
        out[fid] = (y.values.astype(int), p)
    return out


def _probs_clustered(cluster_results) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Each held-out facility scored by its own care-type cluster ensemble."""
    held = load_held_out()
    out = {}
    for fid, sp in held.items():
        cr = cluster_results[FACILITY_CARE_TYPES[fid]]
        total = sum(n for _, n in cr.values())
        X, y = sp["test"]
        p = np.zeros(len(X))
        for _, (b, n) in cr.items():
            p += deserialize_xgb_model(b).predict_proba(X.values)[:, 1] * (n / total)
        out[fid] = (y.values.astype(int), p)
    return out


def _probs_centralised() -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    model = xgb.XGBClassifier(
        n_estimators=XGB_CFG["n_estimators"], max_depth=XGB_CFG["max_depth"],
        learning_rate=XGB_CFG["learning_rate"], verbosity=0,
    )
    model.load_model(str(LOGS_DIR / "centralised_model.json"))
    held = load_held_out()
    return {fid: (sp["test"][1].values.astype(int),
                  model.predict_proba(sp["test"][0].values)[:, 1])
            for fid, sp in held.items()}


def _pool(per_fac):
    y = np.concatenate([per_fac[f][0] for f in sorted(per_fac)])
    p = np.concatenate([per_fac[f][1] for f in sorted(per_fac)])
    return y, p


def _boot_auc(y, probs_by_strat: Dict[str, np.ndarray], n_boot=N_BOOT):
    """Paired bootstrap: same resampled indices for every strategy."""
    rng = np.random.default_rng(SEED)
    n = len(y)
    samples = {s: [] for s in probs_by_strat}
    deltas = []
    done = 0
    while done < n_boot:
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue  # single-class resample -- AUC undefined; redraw
        aucs = {s: roc_auc_score(y[idx], p[idx]) for s, p in probs_by_strat.items()}
        for s, a in aucs.items():
            samples[s].append(a)
        if "clustered_fl" in aucs and "fedavg" in aucs:
            deltas.append(aucs["clustered_fl"] - aucs["fedavg"])
        done += 1
    out = {}
    for s, arr in samples.items():
        arr = np.array(arr)
        out[s] = {
            "auc_point": round(float(roc_auc_score(y, probs_by_strat[s])), 4),
            "auc_boot_mean": round(float(arr.mean()), 4),
            "ci95": [round(float(np.percentile(arr, 2.5)), 4),
                     round(float(np.percentile(arr, 97.5)), 4)],
        }
    d = np.array(deltas)
    delta = {
        "point": round(float(roc_auc_score(y, probs_by_strat["clustered_fl"])
                             - roc_auc_score(y, probs_by_strat["fedavg"])), 4),
        "boot_mean": round(float(d.mean()), 4),
        "ci95": [round(float(np.percentile(d, 2.5)), 4),
                 round(float(np.percentile(d, 97.5)), 4)],
        "prop_positive": round(float((d > 0).mean()), 4),
    }
    return out, delta


def main():
    logger.info("Running 50-round simulations (deterministic) ...")
    fedavg_res = _run_fl_50rounds("fedavg")
    cluster_res = _run_fl_50rounds("clustered")

    pf = {
        "fedavg": _probs_global(fedavg_res),
        "clustered_fl": _probs_clustered(cluster_res),
        "centralised": _probs_centralised(),
    }

    # ── overall (pooled facilities 6 + 9) ─────────────────────────────────────
    y_pool, _ = _pool(pf["fedavg"])
    probs_pool = {s: _pool(v)[1] for s, v in pf.items()}
    overall, delta_overall = _boot_auc(y_pool, probs_pool)

    # ── per held-out facility ────────────────────────────────────────────────
    per_fac = {}
    for fid in sorted(pf["fedavg"]):
        y = pf["fedavg"][fid][0]
        probs = {s: v[fid][1] for s, v in pf.items()}
        strat_ci, delta = _boot_auc(y, probs)
        per_fac[str(fid)] = {
            "care_type": FACILITY_CARE_TYPES[fid],
            "n_instances": int(len(y)), "n_positive": int(y.sum()),
            "strategies": strat_ci, "delta_cfl_minus_fedavg": delta,
        }

    result = {
        "method": (f"Paired instance-level bootstrap, {N_BOOT} resamples, seed {SEED}. "
                   "Same resampled indices applied to every strategy (errors correlated). "
                   "Pipeline is deterministic across rounds; uncertainty quantified over "
                   "held-out test instances, not communication rounds."),
        "n_pooled_instances": int(len(y_pool)),
        "n_pooled_positive": int(y_pool.sum()),
        "overall": {"strategies": overall, "delta_cfl_minus_fedavg": delta_overall},
        "per_facility": per_fac,
    }
    out_path = TABLES_DIR / "bootstrap_ci.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved {out_path}")

    print(f"\n-- Paired bootstrap ({N_BOOT} resamples, n={len(y_pool)} pooled instances, "
          f"{int(y_pool.sum())} positive) --")
    for s, v in overall.items():
        print(f"  {s:<14} AUC {v['auc_point']:.4f}  95% CI [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]")
    d = delta_overall
    print(f"  DELTA CFL-FedAvg: {d['point']:+.4f}  95% CI [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  "
          f"P(delta>0)={d['prop_positive']:.3f}")
    for fid, v in per_fac.items():
        dd = v["delta_cfl_minus_fedavg"]
        print(f"  facility {fid} ({v['care_type']}): delta {dd['point']:+.4f}  "
              f"CI [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  P(>0)={dd['prop_positive']:.3f}")


if __name__ == "__main__":
    main()
