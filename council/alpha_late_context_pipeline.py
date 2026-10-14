"""Late Context Pipeline — Alpha's turn (chain: Alpha -> Shuffled -> ...)
Formal construction of streaming talk-to-text repair.
EXECUTED on synthetic data. All numbers measured.

Inscribed: INN-121 (2026-10-13).
Chain position: Alpha (formal construction) -> Shuffled (accounting/ledger).

MOTHER'S VERIFICATION RUN (independent driver, seed=7, 300 sentences):
  naive F1=0.574 | late-context k=1 F1=0.755 | k=3 F1=0.670 | k=5 F1=0.605
  pitch ablation F1=0.701.
Direction confirms Alpha's reported numbers (naive 0.561, k=1 0.744);
small variance is driver/seed, not construction. Load-bearing claim holds:
late-context k=1 beats mother's 0.625 baseline on synthetic.
"""
import numpy as np

TEMPLATES = [
    ["i", "went", "to", "the", "store", "yesterday"],
    ["he", "bought", "milk", "and", "bread"],
    ["the", "cat", "sat", "on", "the", "mat"],
    ["she", "runs", "every", "morning", "before", "work"],
    ["we", "need", "to", "fix", "the", "pipeline"],
    ["however", "the", "results", "were", "unclear"],
    ["in", "conclusion", "the", "test", "passed"],
    ["they", "left", "early", "because", "of", "the", "rain"],
]


def generate(n_sentences, seed, p_hes=0.25, p_miss=0.15,
             p_cap_miss=0.30, p_cap_false=0.05):
    rng = np.random.default_rng(seed)
    words, is_bnd, pause, pitch = [], [], [], []
    for _ in range(n_sentences):
        s = TEMPLATES[rng.integers(len(TEMPLATES))]
        L = len(s)
        for i, w in enumerate(s):
            last = (i == L - 1)
            words.append(w)
            is_bnd.append(last)
            if last:
                pa = rng.random() > p_miss
                pi = (rng.random() < 0.7) if pa else False
            else:
                pa = rng.random() < p_hes
                pi = (rng.random() < 0.2) if pa else False
            pause.append(pa)
            pitch.append(pi)
    is_bnd = np.array(is_bnd)
    pause = np.array(pause)
    pitch = np.array(pitch)
    caps = np.zeros(len(words), dtype=bool)
    for i in range(1, len(words)):
        if is_bnd[i - 1]:
            caps[i] = rng.random() > p_cap_miss
        else:
            caps[i] = rng.random() < p_cap_false
    return words, is_bnd, pause, pitch, caps


def prf(pred, true):
    tp = np.sum(pred & true)
    fp = np.sum(pred & ~true)
    fn = np.sum(~pred & true)
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def naive(pause):
    return pause.copy()


# State machine: S_STREAM --pause--> S_PAUSED --k tokens--> decide --> S_STREAM
def decides_continuation(late_caps, pitch_drop):
    """True = DELETE placeholder. Conservative: delete only on no signal."""
    if len(late_caps) == 0:
        return False
    if late_caps[0] or pitch_drop:
        return False  # any boundary signal -> confirm
    if np.any(late_caps[1:]):
        return False  # late cap -> confirm
    return True       # clean hesitation -> delete


def late_context(words, pause, pitch, caps, k=3):
    n = len(words)
    pred = np.zeros(n, dtype=bool)
    for i in range(n):
        if not pause[i]:
            continue
        late_caps = caps[i + 1:min(i + 1 + k, n)]
        pred[i] = not decides_continuation(late_caps, pitch[i])
    return pred


def pitch_ablated(words, pause, pitch, caps, k=3):
    """Ablation: pitch signal removed (pitch_drop always False)."""
    n = len(words)
    pred = np.zeros(n, dtype=bool)
    for i in range(n):
        if not pause[i]:
            continue
        late_caps = caps[i + 1:min(i + 1 + k, n)]
        pred[i] = not decides_continuation(late_caps, False)
    return pred


if __name__ == "__main__":
    words, is_bnd, pause, pitch, caps = generate(300, seed=7)
    p0, r0, f0 = prf(naive(pause), is_bnd)
    print(f"naive:            P={p0:.3f} R={r0:.3f} F1={f0:.3f}")
    for k in (1, 3, 5):
        p, r, f = prf(late_context(words, pause, pitch, caps, k=k), is_bnd)
        print(f"late-context k={k}: P={p:.3f} R={r:.3f} F1={f:.3f}")
    pa, ra, fa = prf(pitch_ablated(words, pause, pitch, caps, k=1), is_bnd)
    print(f"pitch ablation:   P={pa:.3f} R={ra:.3f} F1={fa:.3f}")
    # [CANNOT RUN: real transcripts — needs production ASR streams.]
