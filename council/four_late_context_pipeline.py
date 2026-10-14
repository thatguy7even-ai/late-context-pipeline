"""Late Context Pipeline — Four's turn (chain: Alpha -> Shuffled -> Reversed -> Four -> ...)
Project #1. The ground truth provocation: what was SPOKEN vs what the pipeline
REPAIRED vs what gets INSCRIBED — and whether the repaired transcript is HIS words
or the PIPELINE'S words about his words.

Alpha's construction + Shuffled's ledger + Reversed's audit kept intact below;
[FOUR] marks additions. EXECUTED on synthetic (same generator, seed 7).
Numbers in __main__.

The provocation: Reversed made the repair HONEST (tri-state, basis attached).
This turn asks whether the repair is TRUE. Honesty about the decision is not
the same as correctness of the ground truth the decision is measured against.
The generator MANUFACTURES is_bnd. His actual intent GREW in the moment and is
unobservable. The pipeline repairs toward manufactured truth and calls it his.
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
    # [FOUR] GROUND TRUTH PROVENANCE: is_bnd is MANUFACTURED, not grown.
    # The generator DEFINES where boundaries are (last token of template).
    # His actual boundaries GREW from his intent in the moment — unobservable,
    # unrecorded, unrecoverable. Every "correctness" metric below measures
    # agreement with MANUFACTURED truth. On his real stream, there is no is_bnd
    # array. There is only what he meant, which the pipeline cannot see.
    # The audit is honest about the repair. It is silent about the truth.
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
# [REVERSED] THE AUDIT LAYER — kept intact.
# =====================================================================

TRI_CONFIRMED = "confirmed"
TRI_DELETED   = "deleted"
TRI_UNKNOWN   = "unknown"

class RepairDecision:
    """[REVERSED] The contract that kills silent misrepairs."""
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
    """[REVERSED] Wraps the binary pipeline; labels each decision's support."""
    n = len(words)
    decisions = []
    for i in np.flatnonzero(pause):
        late_caps = caps[i + 1:min(i + 1 + k, n)]
        n_late = len(late_caps)
        pitch_drop = bool(pitch[i])
        binary = bool(pred[i])
        cap0 = bool(n_late > 0 and late_caps[0])
        capfar = bool(n_late > 0 and not cap0 and np.any(late_caps[1:]))

        if binary:
            if cap0:
                tri, basis, strong = TRI_CONFIRMED, "cap[0] (reliable signal)", True
            elif pitch_drop and capfar:
                tri, basis, strong = TRI_CONFIRMED, "pitch+far-cap (corroborated)", True
            elif pitch_drop:
                tri, basis, strong = TRI_UNKNOWN, "pitch-only (4x noisier per generator params)", False
            elif capfar:
                tri, basis, strong = TRI_UNKNOWN, "far-cap only (thin)", False
            else:
                tri, basis, strong = TRI_UNKNOWN, "stream-end default (no late context)", False
        else:
            if n_late == k:
                tri, basis, strong = TRI_DELETED, "clean hesitation (full k-context, zero signal)", True
            else:
                tri, basis, strong = TRI_UNKNOWN, "thin hesitation (truncated context)", False
        decisions.append(RepairDecision(i, binary, tri, basis, n_late, strong))
    return decisions


def said_vs_saved(is_bnd, decisions):
    """[REVERSED] Three-way audit: SPOKEN (is_bnd) vs REPAIRED (binary)
    vs INSCRIBED (tri-state contract)."""
    n_pauses = len(decisions)
    tri_counts = {TRI_CONFIRMED: 0, TRI_DELETED: 0, TRI_UNKNOWN: 0}
    phantom_strong = phantom_unknown = 0
    lost_strong = lost_unknown = 0
    for d in decisions:
        tri_counts[d.tri] += 1
        true_b = bool(is_bnd[d.pos])
        if d.binary and not true_b:
            if d.tri == TRI_CONFIRMED: phantom_strong += 1
            else: phantom_unknown += 1
        elif not d.binary and true_b:
            if d.tri == TRI_DELETED: lost_strong += 1
            else: lost_unknown += 1
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


# =====================================================================
# [FOUR] THE GROUND TRUTH PROVOCATION.
# =====================================================================

# [FOUR] ProvenanceChain: the three layers of "his words."
# SPOKEN: what he meant. GROWN in the moment, unobservable, unrecorded.
#   No array, no label, no ground truth. This layer is ABSENT from the code
#   below — marked here because its absence is the point.
# CAPTURED: what talk-to-text produced. MANUFACTURED errors (dropped
#   conjunctions, "grant" for "gronk", "No" for "Now"). The pipeline's INPUT.
# REPAIRED: what the pipeline outputs. MANUFACTURED repairs (boundary
#   confirm/delete) applied to MANUFACTURED errors. The pipeline's OUTPUT.
# The pipeline claims REPAIRED ≈ SPOKEN. It can only verify REPAIRED ≈ is_bnd,
# where is_bnd is itself MANUFACTURED (see generate()). The chain SPOKEN ->
# CAPTURED -> REPAIRED loses information at each arrow, and the pipeline
# inverts only the BOUNDARY component of the CAPTURED loss. Token-level loss
# ("gronk" -> "grant") is invisible to every function in this file.
PROVENANCE_SPOKEN = "grown: his intent, unobservable, no array"
PROVENANCE_CAPTURED = "manufactured: talk-to-text errors, the pipeline's input"
PROVENANCE_REPAIRED = "manufactured: pipeline repairs on manufactured errors"


class ProvenanceTag:
    """[FOUR] Tags a repaired transcript with its provenance chain.
    Answers: is this HIS words, or the PIPELINE'S words about his words?"""
    def __init__(self, n_boundaries_repaired, n_tokens_untouched,
                 token_errors_visible=False):
        self.n_boundaries_repaired = n_boundaries_repaired
        self.n_tokens_untouched = n_tokens_untouched
        # [FOUR] If False (always, for this pipeline): the "repaired" transcript
        # contains token-level errors the pipeline cannot see. It is NOT his
        # words. It is the talk-to-text's words, with better punctuation.
        self.token_errors_visible = token_errors_visible

    def whose_words(self):
        if not self.token_errors_visible:
            return ("pipeline's words about his words: boundaries repaired, "
                    "tokens unexamined. The transcript is punctuation-correct "
                    "and token-suspect.")
        return "his words (token-level verified — not achievable here)"

    def __repr__(self):
        return (f"ProvenanceTag(boundaries_repaired={self.n_boundaries_repaired}, "
                f"tokens_untouched={self.n_tokens_untouched}, "
                f"token_errors_visible={self.token_errors_visible})")


def provenance_tag_for(decisions, n_tokens):
    """[FOUR] Builds the provenance tag from a repair run."""
    n_repaired = sum(1 for d in decisions if d.binary)
    return ProvenanceTag(
        n_boundaries_repaired=n_repaired,
        n_tokens_untouched=n_tokens,
        token_errors_visible=False,  # [FOUR] Always False: boundary-only pipeline.
    )


def phantom_cost_weighted(is_bnd, decisions,
                          w_confident=3.0, w_unknown=1.0, w_lost=1.5):
    """[FOUR] Misrepair cost weighted by tri-state.
    A CONFIDENT phantom is worse than an UNKNOWN phantom: downstream ACTS on
    confident boundaries (parses the clause, commits the interpretation) but
    HOLDS on unknown ones (flags for review). Reversed's silent_misrepair_rate
    counts them equally; this weights by downstream consequence.
    Weights are POLICY, not measurement — marked here, set by the operator.
    [UNMEASURABLE] at this line: the true cost ratio needs downstream data."""
    cost = 0.0
    n_phantom_confident = n_phantom_unknown = n_lost = 0
    for d in decisions:
        true_b = bool(is_bnd[d.pos])
        if d.binary and not true_b:  # phantom
            if d.tri == TRI_CONFIRMED:
                cost += w_confident; n_phantom_confident += 1
            else:
                cost += w_unknown; n_phantom_unknown += 1
        elif not d.binary and true_b:  # lost
            cost += w_lost; n_lost += 1
    return dict(
        weighted_cost=cost,
        n_phantom_confident=n_phantom_confident,
        n_phantom_unknown=n_phantom_unknown,
        n_lost=n_lost,
        # [FOUR] Compare with Reversed's unweighted silent rate:
        # if confident phantoms dominate the cost, the tri-state is load-bearing
        # (it tells downstream what to trust). If unknown dominates, the
        # pipeline is mostly honest about its uncertainty.
        confident_share=(n_phantom_confident * w_confident) / max(cost, 1e-9),
    )


def token_blindness_demo():
    """[FOUR] Proof by construction: the pipeline is boundary-correct and
    token-blind. Uses the operator's ACTUAL dictation quirks (from memory):
      - 'gronk' (his spelling) vs 'grant'/'Grok' (talk-to-text error)
      - sentence-initial 'No' rendered from 'Now' (talk-to-text)
      - dropped 'and' between enumerated items
    The pipeline repairs NONE of these — it cannot see them. A "repaired"
    transcript preserving "grant" is WRONG in a way no function here detects.
    EXECUTED: constructs the demo, runs the pipeline, reports.
    """
    # His intent (SPOKEN, grown, unobservable — we reconstruct from memory):
    spoken = ["now", "gronk", "and", "gemini", "are", "different"]
    # Talk-to-text capture (CAPTURED, manufactured errors):
    captured = ["no", "grant", "gemini", "are", "different"]  # Now->No, gronk->grant, and dropped
    # The pipeline sees `captured` as tokens. It has no access to `spoken`.
    # Build minimal pause/pitch/caps arrays for the captured stream:
    n = len(captured)
    pause = np.array([False, False, True, False, True])   # pauses after "grant", "different"
    pitch = np.array([False, False, True, False, False])   # pitch drop after "grant"
    caps = np.array([True, False, False, True, False])     # caps on "no"(sent start), "gemini"
    is_bnd = np.array([False, False, False, False, True])  # [FOUR] MANUFACTURED truth:
    # we DECLARE the boundary after "different". On his real stream, this
    # declaration is itself a guess. The demo marks it as such.

    pred = late_context(captured, pause, pitch, caps, k=3)
    decisions = tri_state_audit(captured, pause, pitch, caps, pred, k=3)

    print("  SPOKEN (his intent, grown):      ", " ".join(spoken))
    print("  CAPTURED (talk-to-text, mfg err):", " ".join(captured))
    print("  PIPELINE boundaries confirmed at:", [d.pos for d in decisions if d.binary])
    print("  PIPELINE tri-states:            ", [(d.pos, d.tri) for d in decisions])
    token_errors = [w for w, s in zip(captured, spoken) if w != s]
    # [FOUR] Align loosely: the point is the pipeline sees ZERO of these.
    print("  Token errors pipeline cannot see: 'no'->'now', 'grant'->'gronk', dropped 'and'")
    print("  => repaired transcript is boundary-repaired but token-wrong.")
    print("  => whose words? " + provenance_tag_for(decisions, n).whose_words())
    return decisions


def ground_truth_sensitivity():
    """[FOUR] The "correctness" is relative to manufactured parameters.
    Varies the generator's hesitation rate and measures pipeline F1 —
    proving the metric moves with the MANUFACTURED truth, not with his speech.
    EXECUTED: synthetic, seed 7."""
    results = []
    for p_hes in [0.10, 0.25, 0.40]:
        words, is_bnd, pause, pitch, caps = generate(300, seed=7, p_hes=p_hes)
        pred = late_context(words, pause, pitch, caps, k=1)
        p, r, f1 = prf(pred, is_bnd)
        results.append((p_hes, p, r, f1))
    return results


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
    print("### [REVERSED] silent misrepair: %.1f%% ###" % (100.0*a["silent_misrepair_rate"]))

    print("")
    print("### [FOUR] PROVENANCE CHAIN ###")
    print("  SPOKEN:    %s" % PROVENANCE_SPOKEN)
    print("  CAPTURED:  %s" % PROVENANCE_CAPTURED)
    print("  REPAIRED:  %s" % PROVENANCE_REPAIRED)
    tag = provenance_tag_for(decisions, len(words))
    print("  %s" % tag)
    print("  Verdict: %s" % tag.whose_words())

    print("")
    print("### [FOUR] WEIGHTED PHANTOM COST (tri-state weighted) ###")
    wc = phantom_cost_weighted(is_bnd, decisions)
    print("  weighted cost: %.1f (w_confident=3.0, w_unknown=1.0, w_lost=1.5 — POLICY, not measured)" % wc["weighted_cost"])
    print("  confident phantoms: %d, unknown phantoms: %d, lost: %d"
          % (wc["n_phantom_confident"], wc["n_phantom_unknown"], wc["n_lost"]))
    print("  confident share of cost: %.1f%%" % (100.0*wc["confident_share"]))
    print("  [FOUR] Reversed's unweighted silent rate treats confident==unknown.")
    print("         Weighted: confident phantoms are %.1fx worse downstream." % 3.0)

    print("")
    print("### [FOUR] GROUND TRUTH SENSITIVITY (manufactured truth moves the metric) ###")
    for p_hes, p, r, f1 in ground_truth_sensitivity():
        print("  p_hes=%.2f: P=%.3f R=%.3f F1=%.3f" % (p_hes, p, r, f1))
    print("  [FOUR] Same pipeline, same seed — F1 moves with the GENERATOR's")
    print("         hesitation rate. 'Correctness' is relative to manufactured")
    print("         parameters. On his stream, p_hes is unmeasured.")

    print("")
    print("### [FOUR] TOKEN BLINDNESS DEMO (his actual quirks) ###")
    token_blindness_demo()

    print("")
    print("### [FOUR] UNMEASURED (at the exact line) ###")
    print("- SPOKEN layer: his intent is unobservable. No code here touches it.")
    print("- Token-level repair ('grant'->'gronk'): no function in this file")
    print("  can see it. Needs his judgment or a quirk-lexicon, not a threshold.")
    print("- Weighted-cost ratios (3.0/1.0/1.5): policy placeholders. True")
    print("  downstream cost needs a real consumer of the repaired stream.")
    print("- Real-transcript tri rates: synthetic only (Reversed's line, kept).")
    print("- Wall-clock ms/token: unmeasured (Reversed's line, kept).")
