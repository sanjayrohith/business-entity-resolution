"""Minimal inference helpers for the two LightGBM models in this repository.

The models score *candidate pairs* (S1 query record, S2/S3 target record) that were
already produced by the retrieval stage and turned into the 80 pair features of the
full pipeline: https://github.com/sanjayrohith/business-entity-resolution

Pipeline implemented here:
    80 pair features --base model--> base score
    base score + 41 base features + 14 per-S1 context features --meta model--> meta score
    meta score >= 0.70 --> predicted link
    target claimed by several S1 --> keep the owner only if it leads by >= 0.05
"""
import json
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np

HERE = Path(__file__).resolve().parent
BASE_THRESHOLD = 0.642   # submission 1 (base model only)
META_THRESHOLD = 0.70    # submissions 2 and 3
OWNERSHIP_MARGIN = 0.05  # submission 3

# Columns of the 80-feature base matrix that the meta model reuses (order matters).
BASE_COLS = [0, 1, 2, 3, 4, 5, 6, 7, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
             30, 31, 32, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 77, 78, 79]
CONTEXT_NAMES = ['base_score', 's1_rank_all', 's1_rank_source', 's1_top_score', 's1_second_score',
                 's1_score_gap_top', 's1_score_gap_second', 's1_count_score_ge_05',
                 's1_count_score_ge_0642', 's1_count_score_ge_08', 's1_count_score_ge_095',
                 'other_source_top_score', 'same_source_top_score', 'source3_indicator']


def load_models(folder=HERE):
    folder = Path(folder)
    base = lgb.Booster(model_file=str(folder / 'base_lightgbm.txt'))
    meta = lgb.Booster(model_file=str(folder / 'meta_lightgbm.txt'))
    features = json.loads((folder / 'base_model_selection.json').read_text(encoding='utf-8'))['features']
    return base, meta, features


def context(q, source, score):
    """Per-S1 score context. q: S1 index per pair, source: 2 or 3 per pair, score: base score."""
    q = np.asarray(q)
    source = np.asarray(source)
    score = np.asarray(score, dtype=np.float32)
    n = len(score)
    ctx = np.zeros((n, len(CONTEXT_NAMES)), dtype=np.float32)
    ctx[:, 0] = score
    ctx[:, -1] = source == 3
    order = np.argsort(q, kind='stable')
    bounds = np.r_[0, np.flatnonzero(np.diff(q[order])) + 1, n]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        ix = order[lo:hi]
        v = score[ix]
        ctx[ix, 1] = np.argsort(np.argsort(-v, kind='stable'), kind='stable') + 1
        top = np.sort(v)[::-1]
        second = top[1] if len(top) > 1 else 0
        ctx[ix, 3] = top[0]
        ctx[ix, 4] = second
        ctx[ix, 5] = top[0] - v
        ctx[ix, 6] = v - second
        for col, t in ((7, .5), (8, .642), (9, .8), (10, .95)):
            ctx[ix, col] = np.count_nonzero(v >= t)
        for src in (2, 3):
            mask = source[ix] == src
            if not mask.any():
                continue
            j = ix[mask]
            vv = score[j]
            ctx[j, 2] = np.argsort(np.argsort(-vv, kind='stable'), kind='stable') + 1
            ctx[j, 12] = vv.max()
            other = score[ix[~mask]]
            ctx[j, 11] = other.max() if len(other) else 0
    return ctx


def score_pairs(X80, q, source, base_model, meta_model):
    """Return (base_score, meta_score) for every candidate pair."""
    X80 = np.asarray(X80, dtype=np.float32)
    if X80.shape[1] != 80:
        raise ValueError(f'Expected 80 pair features, got {X80.shape[1]}')
    base = base_model.predict(X80)
    meta_input = np.concatenate([X80[:, BASE_COLS], context(q, source, base)], axis=1)
    return base, meta_model.predict(meta_input)


def resolve_ownership(links, margin=OWNERSHIP_MARGIN):
    """links: iterable of (s1_id, target_id, meta_score) above threshold.

    Each target keeps at most one S1 owner: the top scorer if it leads the runner-up
    by at least `margin`, otherwise the target is dropped from all claimants.
    """
    owners = defaultdict(dict)
    for s1, target, s in links:
        owners[target][s1] = float(s)
    kept = defaultdict(list)
    for target, claims in owners.items():
        ranked = sorted(claims.items(), key=lambda item: (-item[1], item[0]))
        if len(ranked) == 1 or ranked[0][1] - ranked[1][1] >= margin:
            kept[ranked[0][0]].append(target)
    return dict(kept)


if __name__ == '__main__':
    # Smoke test with synthetic features: 3 S1 queries, 8 candidates each.
    base_model, meta_model, names = load_models()
    rng = np.random.default_rng(0)
    q = np.repeat(np.arange(3), 8)
    source = np.tile([2, 2, 2, 2, 3, 3, 3, 3], 3)
    X = rng.random((len(q), 80), dtype=np.float32)
    base, meta = score_pairs(X, q, source, base_model, meta_model)
    targets = [f'T{i % 10}' for i in range(len(q))]  # some targets shared between queries
    links = [(f'S1-{qi}', t, s) for qi, t, s in zip(q, targets, meta) if s >= META_THRESHOLD]
    print('features:', len(names), '| base trees:', base_model.num_trees(), '| meta trees:', meta_model.num_trees())
    print('base score range: %.3f-%.3f, meta score range: %.3f-%.3f' % (base.min(), base.max(), meta.min(), meta.max()))
    print('links above threshold:', len(links), '| after ownership:', sum(map(len, resolve_ownership(links).values())))
