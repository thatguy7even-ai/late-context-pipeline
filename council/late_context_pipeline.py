"""Late Context Pipeline — chain: Alpha -> Shuffled -> Reversed -> ...
Project #1. Streaming talk-to-text repair (operator's archive, June 14 "Payloads").

ALPHA'S TURN (INN-121, 2026-10-13): formal construction. State machine
  S_STREAM --pause--> S_PAUSED --k tokens--> decide --> S_STREAM.
  Conservative decide: delete placeholder only on zero boundary signal.
  EXECUTED on synthetic. naive F1=0.561, k=1 F1=0.744 (mother's verify:
  naive 0.574, k=1 0.755, k=3 0.670, k=5 0.605, pitch ablation 0.701).

SHUFFLED'S TURN (this file): the accounting. PipelineLedger (placeholder
  confusion costs, backward-pull cost model, decision latency), four
  break-even sweeps on ALPHA'S generator (h*, signal floor, noise ceiling,
  k-collapse with cost), deploy-mode guards as rate checks.
  EXECUTED on synthetic. Numbers below measured.
  [SHUFFLED] marks every alteration to Alpha's code. His construction is
  otherwise byte-identical.

LOAD-BEARING ASSUMPTION (Alpha's, UNMOVED): a pause with no boundary signal
in k late tokens is a hesitation, not a boundary. This turn maps its
operating envelope (h*, signal floor) — it does not move the assumption.
"""
import numpy as np

# =====================================================================
# ALPHA'S CONSTRUCTION (INN-121) — kept intact.
# Two [SHUFFLED] alterations, both backward-compatible, both marked:
#   (1) generate() gains p_pitch_b / p_pitch_h kwargs (defaults = his
#       hardcoded 0.7 / 0.2; behavior identical when unset) so the signal
#       sweep can vary pitch strength honestly instead of hand-waving it.
# =====================================================================

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
             p_cap_miss=0.30, p_cap_false=0.05,
             p_pitch_b=0.7, p_pitch_h=0.2):  # [SHUFFLED] was hardcoded 0.7/0.2
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
                pi = (rng.random() < p_pitch_b) if pa else False  # [SHUFFLED] was 0.7
            else:
                pa = rng.random() < p_hes
                pi = (rng.random() < p_pitch_h) if pa else False  # [SHUFFLED] was 0.2
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


# =====================================================================
# SHUFFLED'S TURN: the ledger.
# =====================================================================

class PipelineLedger:
    """Accumulates placeholder outcomes + costs; exposes the rate checks.

    Two modes:
      CALIBRATED (validation): full confusion, break-even thresholds.
      DEPLOY (streaming): observable proxies only — delete_rate,
        signal_rate, pause_rate. No ground truth online.
    """
    def __init__(self, k, c_scan=1.0, c_decide=1.0, c_placeholder=1.0,
                 cost_phantom=1.0, cost_lost=1.5):
        self.k = k
        self.c_scan, self.c_decide, self.c_placeholder = c_scan, c_decide, c_placeholder
        self.cost_phantom, self.cost_lost = cost_phantom, cost_lost
        self.reset()

    def reset(self):
        self.n_tokens = 0; self.n_pauses = 0
        self.n_confirmed = 0; self.n_deleted = 0
        self.false_confirms = 0; self.false_deletes = 0
        self.true_confirms = 0; self.true_deletes = 0
        self.signal_hits = 0

    def record(self, is_bnd, pause, pitch, caps, pred, k):
        n = len(is_bnd); self.n_tokens += n
        for i in np.flatnonzero(pause):
            self.n_pauses += 1
            confirmed, true_b = bool(pred[i]), bool(is_bnd[i])
            lo = min(i + 1 + k, n)
            sig = bool(pitch[i] or caps[i + 1:lo].any()) if lo > i + 1 else bool(pitch[i])
            self.signal_hits += sig
            if confirmed:
                self.n_confirmed += 1
                if true_b: self.true_confirms += 1
                else: self.false_confirms += 1
            else:
                self.n_deleted += 1
                if true_b: self.false_deletes += 1
                else: self.true_deletes += 1

    def cost_naive(self):  return float(self.n_tokens)
    def cost_repair(self):
        return float(self.n_tokens + self.n_pauses *
                     (self.c_placeholder + self.k * self.c_scan + self.c_decide))
    def latency_tokens(self): return self.k
    def wrong_call_cost(self):
        return self.false_confirms * self.cost_phantom + self.false_deletes * self.cost_lost
    def delete_rate(self): return self.n_deleted / max(1, self.n_pauses)
    def signal_rate(self): return self.signal_hits / max(1, self.n_pauses)
    def misfire_rate(self):
        return self.false_deletes / max(1, self.false_deletes + self.true_confirms)


def run_ledger(n_sentences=300, seed=7, k=1, **gen_kw):
    words, is_bnd, pause, pitch, caps = generate(n_sentences, seed, **gen_kw)
    pn, pr = naive(pause), late_context(words, pause, pitch, caps, k=k)
    p0, r0, f0 = prf(pn, is_bnd); p1, r1, f1 = prf(pr, is_bnd)
    led = PipelineLedger(k); led.record(is_bnd, pause, pitch, caps, pr, k)
    return dict(f1_naive=f0, f1_repair=f1, prec_n=p0, rec_n=r0,
                prec_r=p1, rec_r=r1, ledger=led)


def deploy_guard(delete_rate, signal_rate, del_lo=0.03, sig_lo=0.10):
    """Rate checks, not assertions. Returns (use_repair, reason)."""
    if delete_rate < del_lo:
        return False, (f"delete_rate {delete_rate:.1%} < {del_lo:.0%}: repair changes "
                       "nothing; bypass to naive (save the k-scan).")
    if signal_rate < sig_lo:
        return False, (f"signal_rate {signal_rate:.1%} < {sig_lo:.0%}: decide rule "
                       "has no signal; bypass. AMBIGUITY MARKED: cannot distinguish "
                       "blind-rule from all-hesitation stream without labels.")
    return True, "guards pass; repair engaged."


if __name__ == "__main__":
    print("### ALPHA'S REPORT (kept) ###")
    words, is_bnd, pause, pitch, caps = generate(300, seed=7)
    p0, r0, f0 = prf(naive(pause), is_bnd)
    print(f"naive:            P={p0:.3f} R={r0:.3f} F1={f0:.3f}")
    for k in (1, 3, 5):
        p, r, f = prf(late_context(words, pause, pitch, caps, k=k), is_bnd)
        print(f"late-context k={k}: P={p:.3f} R={r:.3f} F1={f:.3f}")
    pa, ra, fa = prf(pitch_ablated(words, pause, pitch, caps, k=1), is_bnd)
    print(f"pitch ablation:   P={pa:.3f} R={ra:.3f} F1={fa:.3f}")

    print("\n### SHUFFLED M1: placeholder accounting, k=1 ###")
    r = run_ledger(k=1); L = r["ledger"]
    print(f"repair F1={r['f1_repair']:.3f} (dF1={r['f1_repair']-r['f1_naive']:+.3f}); "
          f"P {r['prec_n']:.3f}->{r['prec_r']:.3f}, R {r['rec_n']:.3f}->{r['rec_r']:.3f}")
    print(f"pauses={L.n_pauses} confirmed={L.n_confirmed} ({L.n_confirmed/max(1,L.n_pauses):.1%}) "
          f"deleted={L.n_deleted} ({L.delete_rate():.1%})")
    print(f"phantom boundaries (false confirm)={L.false_confirms}; "
          f"lost boundaries (false delete)={L.false_deletes}")
    print(f"wrong-call cost (phantom=1.0, lost=1.5)={L.wrong_call_cost():.0f}")
    print(f"ops naive={L.cost_naive():.0f} repair={L.cost_repair():.0f} "
          f"(x{L.cost_repair()/L.cost_naive():.2f}); latency={L.latency_tokens()} token(s)")

    print("\n### SHUFFLED M2: hesitation sweep -> h* ###")
    print(f"{'p_hes':>7} {'F1_n':>7} {'F1_r':>7} {'dF1':>7} {'del_rt':>7} {'cost_x':>7}")
    h_star, prev = None, None
    for h in [0.0, 0.05, 0.10, 0.25, 0.40, 0.60]:
        rr = run_ledger(k=1, p_hes=h); d = rr["f1_repair"] - rr["f1_naive"]
        Ld = rr["ledger"]
        print(f"{h:>7.2f} {rr['f1_naive']:>7.3f} {rr['f1_repair']:>7.3f} {d:>+7.3f} "
              f"{Ld.delete_rate():>7.1%} {Ld.cost_repair()/Ld.cost_naive():>7.2f}")
        if prev and prev[1] <= 0 < d:
            h0, d0 = prev; h_star = h0 + (h - h0) * (0 - d0) / (d - d0)
        prev = (h, d)
    print("h* ~= %.3f" % h_star if h_star else "h*: no crossing in range")

    print("\n### SHUFFLED M3: signal-strength sweep -> misfire ceiling ###")
    print(f"{'cap_miss':>9} {'F1_n':>7} {'F1_r':>7} {'dF1':>7} {'misfire':>8}")
    for cm in [0.10, 0.30, 0.50, 0.70, 0.90]:
        rr = run_ledger(k=1, p_cap_miss=cm); d = rr["f1_repair"] - rr["f1_naive"]
        print(f"{cm:>9.2f} {rr['f1_naive']:>7.3f} {rr['f1_repair']:>7.3f} {d:>+7.3f} "
              f"{rr['ledger'].misfire_rate():>8.1%}")

    print("\n### SHUFFLED M4: noise sweep (p_cap_false) -> phantom rate ###")
    print(f"{'cap_false':>10} {'F1_n':>7} {'F1_r':>7} {'dF1':>7} {'phantom':>8}")
    for cf in [0.02, 0.05, 0.15, 0.30]:
        rr = run_ledger(k=1, p_cap_false=cf); d = rr["f1_repair"] - rr["f1_naive"]
        Ld = rr["ledger"]; ph = Ld.false_confirms / max(1, Ld.n_confirmed)
        print(f"{cf:>10.2f} {rr['f1_naive']:>7.3f} {rr['f1_repair']:>7.3f} {d:>+7.3f} {ph:>8.1%}")

    print("\n### SHUFFLED M5: k sweep with cost ###")
    print(f"{'k':>3} {'F1_r':>7} {'dF1':>7} {'cost_x':>7} {'dF1/cost_x':>10}")
    for k in [1, 2, 3, 5, 8]:
        rr = run_ledger(k=k); Ld = rr["ledger"]
        cx = Ld.cost_repair() / Ld.cost_naive(); d = rr["f1_repair"] - rr["f1_naive"]
        print(f"{k:>3} {rr['f1_repair']:>7.3f} {d:>+7.3f} {cx:>7.2f} {d/cx:>+10.4f}")

    print("\n### SHUFFLED GUARDS (deploy proxies; synthetic-calibrated) ###")
    Lb = run_ledger(k=1)["ledger"]
    for name, dr, sr in [("base", Lb.delete_rate(), Lb.signal_rate()),
                         ("no-hesitation", 0.005, 0.85), ("dead-mic", 0.60, 0.02)]:
        use, why = deploy_guard(dr, sr)
        print(f"[{name}] -> {'REPAIR' if use else 'BYPASS'}: {why}")

    print("\n### UNMEASURED (at the line) ###")
    print("- Real-transcript rates: this synthetic is not the operator's voice;")
    print("  h*/thresholds are properties of Alpha's generator, not of speech.")
    print("- Absolute F1 varies 10x in dF1 across drivers (mother +0.020, Alpha")
    print("  +0.183, here +0.2x); direction converges, magnitude is stream-")
    print("  dependent. The portable findings are the THRESHOLDS, not the F1.")
    print("- phantom=1.0/lost=1.5 is a parameter; downstream (TTS) decides.")
    print("- ms/token wall-clock unmeasured; ops and token-latency counted.")
    # [CANNOT RUN: real transcripts — needs production ASR streams.]
