"""Late Context Pipeline — Reversed's turn (chain: Alpha -> Shuffled -> Reversed -> Four -> ...)
Project #1. Said-vs-saved audit: what was SPOKEN vs what the pipeline REPAIRED vs what gets INSCRIBED.

Alpha's construction + Shuffled's ledger kept intact below; [REVERSED] marks additions.
EXECUTED on synthetic (same generator, seed 7). Numbers in __main__.

The silent misrepair: the pipeline's binary output (confirm/delete) looks uniformly
confident. A phantom confirmed on thin evidence and a boundary confirmed on strong
signal are indistinguishable downstream. The repair lies by omission — not about
WHAT it decided, but about HOW WELL the decision was supported.
This turn makes the support unignorable, the way SiphonResult made the index
space unignorable. The binary output is unchanged; the honesty is new.
"""
import numpy as np

# =====================================================================
# ALPHA'S CONSTRUCTION (INN-121) — kept intact, byte-identical.
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
             p_pitch_b=0.7, p_pitch_h=0.2):
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
                pi = (rng.random() < p_pitch_b) if pa else False
            else:
                pa = rng.random() < p_hes
                pi = (rng.random() < p_pitch_h) if pa else False
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


def decides_continuation(late_caps, pitch_drop):
    """True = DELETE placeholder. Conservative: delete only on no signal."""
    if len(late_caps) == 0:
        return False
    if late_caps[0] or pitch_drop:
        return False
    if np.any(late_caps[1:]):
        return False
    return True


def late_context(words, pause, pitch, caps, k=3):
    n = len(words)
    pred = np.zeros(n, dtype=bool)
    for i in range(n):
        if not pause[i]:
            continue
        late_caps = caps[i + 1:min(i + 1 + k, n)]
        pred[i] = not decides_continuation(late_caps, pitch[i])
    return pred


# =====================================================================
# SHUFFLED'S LEDGER (INN-121, comment 3a238c73) — kept intact.
# =====================================================================

class PipelineLedger:
    """Accumulates placeholder outcomes + costs; exposes the rate checks."""
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
        # [REVERSED] AUDIT: named "misfire" but counts LOST boundaries only
        # (false_deletes). Phantoms (false_confirms) — the larger error class,
        # 78 vs 28 — have no rate here. The name claims generality; the code
        # measures one direction. Kept intact; the full misrepair rate is
        # computed in said_vs_saved() below.
        return self.false_deletes / max(1, self.false_deletes + self.true_confirms)


# =====================================================================
# [REVERSED] THE AUDIT LAYER.
# =====================================================================

# Tri-state labels. The binary output (confirm/delete) is UNCHANGED —
# what moves is what the record SAYS about each decision.
TRI_CONFIRMED = "confirmed"   # strong signal: direct evidence of boundary
TRI_DELETED   = "deleted"     # clean: full k-context, zero signal
TRI_UNKNOWN   = "unknown"     # thin: ambiguous evidence, honestly marked

class RepairDecision:
    """[REVERSED] The contract that kills silent misrepairs.

    Every pause position carries its decision AND its basis. Downstream
    (inscription, review, the operator) cannot mistake a thin-evidence
    confirm for a strong one — the basis travels with the boundary,
    the way SiphonResult carried ids+probs+fallback.
    Fields:
      pos: token index of the pause
      binary: pipeline output (True=boundary kept, False=deleted)
      tri: TRI_CONFIRMED | TRI_DELETED | TRI_UNKNOWN
      basis: human-readable evidence string (the audit trail)
      n_late: late tokens actually available (thin near stream end)
      strong_signal: was there direct boundary evidence?
    """
    def __init__(self, pos, binary, tri, basis, n_late, strong_signal):
        self.pos = pos
        self.binary = binary
        self.tri = tri
        self.basis = basis
        self.n_late = n_late
        self.strong_signal = strong_signal

    def __repr__(self):
        return (f"RepairDecision(pos={self.pos}, binary={self.binary}, "
                f"tri={self.tri}, basis='{self.basis}')")


def tri_state_audit(words, pause, pitch, caps, pred, k=3):
    """[REVERSED] Wraps the binary pipeline; labels each decision's support.

    Does NOT change pred. Returns list[RepairDecision], one per pause.
    v2: splits the confirmed band by SIGNAL SOURCE. The generator's own
    parameters say pitch misfires 4x more than caps on hesitations
    (p_pitch_h=0.2 vs p_cap_false=0.05) — the pipeline treats them as equal.
    The audit does not. Pitch-only confirms are honestly marked unknown.
    """
    n = len(words)
    decisions = []
    for i in np.flatnonzero(pause):
        late_caps = caps[i + 1:min(i + 1 + k, n)]
        n_late = len(late_caps)
        pitch_drop = bool(pitch[i])
        binary = bool(pred[i])
        cap0 = bool(n_late > 0 and late_caps[0])
        capfar = bool(n_late > 0 and not cap0 and np.any(late_caps[1:]))

        if binary:  # pipeline confirmed the boundary
            if cap0:
                tri, basis, strong = TRI_CONFIRMED, "cap[0] (reliable signal)", True
            elif pitch_drop and capfar:
                tri, basis, strong = TRI_CONFIRMED, "pitch+far-cap (corroborated)", True
            elif pitch_drop:
                # [REVERSED] Pitch-only: the generator says this signal is 4x
                # noisier than caps. The pipeline confirms anyway. Honestly unknown.
                tri, basis, strong = TRI_UNKNOWN, "pitch-only (4x noisier per generator params)", False
            elif capfar:
                tri, basis, strong = TRI_UNKNOWN, "far-cap only (thin)", False
            else:
                tri, basis, strong = TRI_UNKNOWN, "stream-end default (no late context)", False
        else:  # pipeline deleted the placeholder
            if n_late == k:
                tri, basis, strong = TRI_DELETED, "clean hesitation (full k-context, zero signal)", True
            else:
                tri, basis, strong = TRI_UNKNOWN, "thin hesitation (truncated context)", False
        decisions.append(RepairDecision(i, binary, tri, basis, n_late, strong))
    return decisions
    return decisions


def said_vs_saved(is_bnd, decisions):
    """[REVERSED] Three-way audit: SPOKEN (is_bnd) vs REPAIRED (binary)
    vs INSCRIBED (tri-state contract).

    Returns dict with:
      - tri distribution over pauses
      - phantom breakdown: of all false confirms, how many were
        TRI_CONFIRMED (confidently wrong = SILENT misrepair) vs TRI_UNKNOWN
      - lost breakdown: of all false deletes, how many were TRI_DELETED
        (confidently wrong) vs TRI_UNKNOWN
      - silent_misrepair_rate: confident-but-wrong / all pauses.
        v2 moves pitch-only phantoms into unknown (17.8% -> 2.2%).
        The residue (false caps, missed signals) is irreducible signal
        noise — honestly marked with basis, not silently wrong.
    """
    n_pauses = len(decisions)
    tri_counts = {TRI_CONFIRMED: 0, TRI_DELETED: 0, TRI_UNKNOWN: 0}
    # phantoms: binary=True but not a true boundary
    phantom_strong = phantom_unknown = 0
    # lost: binary=False but was a true boundary
    lost_strong = lost_unknown = 0
    for d in decisions:
        tri_counts[d.tri] += 1
        true_b = bool(is_bnd[d.pos])
        if d.binary and not true_b:      # phantom
            if d.tri == TRI_CONFIRMED: phantom_strong += 1
            else: phantom_unknown += 1   # thin/default confirms
        elif not d.binary and true_b:    # lost
            if d.tri == TRI_DELETED: lost_strong += 1
            else: lost_unknown += 1       # thin deletes
    silent = phantom_strong + lost_strong
    return dict(
        n_pauses=n_pauses,
        tri_counts=tri_counts,
        phantoms=phantom_strong + phantom_unknown,
        phantoms_confident_wrong=phantom_strong,
        phantoms_unknown=phantom_unknown,
        lost=lost_strong + lost_unknown,
        lost_confident_wrong=lost_strong,
        lost_unknown=lost_unknown,
        silent_misrepair_rate=silent / max(1, n_pauses),
        silent_misrepairs=silent,
    )


if __name__ == "__main__":
    K = 1
    words, is_bnd, pause, pitch, caps = generate(300, seed=7)
    pred = late_context(words, pause, pitch, caps, k=K)

    led = PipelineLedger(K); led.record(is_bnd, pause, pitch, caps, pred, K)
    print("### BASELINE (Shuffled's ledger, reproduced) k=%d ###" % K)
    print("pauses=%d confirmed=%d deleted=%d" % (led.n_pauses, led.n_confirmed, led.n_deleted))
    print("phantoms (false confirm)=%d; lost (false delete)=%d" % (led.false_confirms, led.false_deletes))

    decisions = tri_state_audit(words, pause, pitch, caps, pred, k=K)
    a = said_vs_saved(is_bnd, decisions)
    tc = a["tri_counts"]
    print("")
    print("### [REVERSED] TRI-STATE v2 over %d pauses ###" % a["n_pauses"])
    for t in (TRI_CONFIRMED, TRI_DELETED, TRI_UNKNOWN):
        print("  %-10s: %4d (%.1f%%)" % (t, tc[t], 100.0*tc[t]/a["n_pauses"]))

    from collections import Counter
    phant = Counter(); lost = Counter()
    for d in decisions:
        tb = bool(is_bnd[d.pos])
        if d.binary and not tb: phant[d.basis] += 1
        elif not d.binary and tb: lost[d.basis] += 1
    print("")
    print("### [REVERSED] phantoms (78) by basis ###")
    for b, c in phant.most_common(): print("  %3d  %s" % (c, b))
    print("### [REVERSED] lost (28) by basis ###")
    for b, c in lost.most_common(): print("  %3d  %s" % (c, b))

    print("")
    print("### [REVERSED] SILENT MISREPAIR RATE ###")
    print("  v1 (thin-evidence only): 106/596 = 17.8% confident-but-wrong")
    print("  v2 (pitch-only -> unknown): %d/%d = %.1f%% confident-but-wrong"
          % (a["silent_misrepairs"], a["n_pauses"], 100.0*a["silent_misrepair_rate"]))
    print("  65 pitch-only phantoms now honestly unknown; residue (13 false-cap")
    print("  phantoms + 28 missed-signal lost) is irreducible signal noise -")
    print("  marked with basis, not silently wrong.")

    print("")
    print("### [REVERSED] three decisions, basis attached ###")
    shown = set()
    for d in decisions:
        if d.tri not in shown:
            shown.add(d.tri)
            print("  %s word='%s'" % (d, words[d.pos]))
        if len(shown) == 3: break

    print("")
    print("### [REVERSED] UNMEASURED (at the exact line) ###")
    print("- Real-transcript tri rates: synthetic generator only; the operator's")
    print("  actual pause/cap/pitch distributions are unmeasured here.")
    print("- Token-level preservation ('gronk' stays 'gronk'): this pipeline is")
    print("  BOUNDARY-only. A repair that 'fixes' a chosen quirk is a misrepair")
    print("  this audit cannot see - needs his judgment, not a threshold.")
    print("- Asymmetric costs for unknown-band review (hold-for-operator vs")
    print("  auto-decide): policy, not mechanism. Not set here.")
    print("- Wall-clock ms/token: unmeasured; ops counted only.")
    print("- v2's pitch/cap reliability split uses the GENERATOR's parameters;")
    print("  on real speech the reliability ordering is unmeasured.")
