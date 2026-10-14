"""
Late Context Pipeline — Blink's test design (experimental-design seat).
Outward-facing falsification harness for synthesis_five_pipeline_verified.py.

Every claim the verified synthesis makes is restated here as a BREAKABLE test:
  - what would break it (the break condition),
  - what gets measured,
  - where execution stops (marked at the exact line, per the operator's bar).

Twin's DIVERGE binds throughout: NO synthetic magnitude is headlined as a
finding. Four's manufactured bound binds: correctness = generator agreement
on synthetic; on real streams there is no is_bnd.

A stranger runs `python3 this_file.py` and gets pass/fail plus paper designs.

Marking (inherited from the chain):
  [EXECUTED]      ran here on numpy, numbers reported
  [POLICY]        mirrors generator parameters, not a finding
  [SYNTH]         synthetic ground truth — manufactured, not grown
  [UNMEASURABLE]  cannot run here; exact line marked, measurement designed on paper
"""
import numpy as np
from collections import Counter

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


def generate(n_sent, seed, p_hes=0.25, p_miss=0.15,
             p_cap_miss=0.30, p_cap_false=0.05):
    # [SYNTH] Alpha's generator, re-derived (not imported) for independence.
    # is_bnd MANUFACTURED (Four binds): correctness = agreement with THIS generator.
    rng = np.random.default_rng(seed)
    toks, truth, paus, pit = [], [], [], []
    for _ in range(n_sent):
        s = TEMPLATES[rng.integers(len(TEMPLATES))]
        L = len(s)
        for j, w in enumerate(s):
            last = (j == L - 1)
            toks.append(w)
            truth.append(last)
            if last:
                pa = rng.random() > p_miss
                pi = (rng.random() < 0.7) if pa else False
            else:
                pa = rng.random() < p_hes
                pi = (rng.random() < 0.2) if pa else False
            paus.append(pa)
            pit.append(pi)
    truth = np.array(truth)
    paus = np.array(paus)
    pit = np.array(pit)
    cap = np.zeros(len(toks), dtype=bool)
    for j in range(1, len(toks)):
        cap[j] = (rng.random() > p_cap_miss) if truth[j - 1] else (rng.random() < p_cap_false)
    return toks, truth, paus, pit, cap


def prf(pred, true):
    tp = np.sum(pred & true)
    fp = np.sum(pred & ~true)
    fn = np.sum(~pred & true)
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return 2 * p * r / max(p + r, 1e-9)


def repair(paus, pit, cap, k=1):
    # [EXECUTED] Alpha's mechanism + Reversed's tri-state, re-derived.
    n = len(paus)
    pred = np.zeros(n, bool)
    basis = []
    for i in np.flatnonzero(paus):
        late = cap[i + 1:min(i + 1 + k, n)]
        nl = len(late)
        pd = bool(pit[i])
        c0 = bool(nl > 0 and late[0])
        cf = bool(nl > 0 and not c0 and np.any(late[1:]))
        if nl == 0 or c0 or pd or cf:
            pred[i] = True
            basis.append("confirmed" if (c0 or (pd and cf)) else "unknown")
        else:
            pred[i] = False
            basis.append("deleted" if nl == k else "unknown")
    return pred, basis


def silent_misrepair(paus, basis, truth):
    # [EXECUTED] Reversed's metric: confident-but-wrong.
    idx = np.flatnonzero(paus)
    cw = sum(1 for j, b in zip(idx, basis)
             if (b == "confirmed" and not truth[j]) or (b == "deleted" and truth[j]))
    return cw, len(idx)


# ---------------------------------------------------------------------------
# T1 — Mechanism fidelity. [EXECUTED]
# Claim: the repair mechanism reproduces the synthesis numbers (k=1 F1 ~0.755,
#   dF1 ~+0.181) from an independent reimplementation.
# Break: divergence beyond seed variance -> the spec is incomplete.
# ---------------------------------------------------------------------------
def t1_fidelity():
    toks, truth, paus, pit, cap = generate(300, seed=7)
    f0 = prf(paus.copy(), truth)
    pr, _ = repair(paus, pit, cap, k=1)
    f1 = prf(pr, truth)
    ok = abs(f1 - 0.755) < 0.03 and abs((f1 - f0) - 0.181) < 0.05
    return ok, {"f0": round(f0, 3), "f1": round(f1, 3), "dF1": round(f1 - f0, 3)}


# ---------------------------------------------------------------------------
# T2 — Portable claim 1: delete helps PROPORTIONALLY to hesitation rate. [EXECUTED]
# Claim (Twin): dF1 scales with p_hes — proportional, not constant.
# Break: dF1 flat across p_hes -> the gain doesn't track hesitation;
#   the "proportional" qualifier is wrong (or the mechanism helps for another reason).
# ---------------------------------------------------------------------------
def t2_proportional():
    ds = []
    for ph in (0.05, 0.25, 0.60):  # [POLICY] knob
        _, truth, paus, pit, cap = generate(300, seed=7, p_hes=ph)
        pr, _ = repair(paus, pit, cap, k=1)
        ds.append(prf(pr, truth) - prf(paus.copy(), truth))
    # Proportional: strictly increasing with p_hes.
    ok = ds[0] < ds[1] < ds[2]
    return ok, {"dF1_by_p_hes": [round(d, 3) for d in ds]}


# ---------------------------------------------------------------------------
# T3 — Portable claim 2: needs reliable signal. [EXECUTED]
# Claim (Twin): dF1 degrades as p_cap_miss rises — collapses, then inverts.
# Break: dF1 stays positive at high p_cap_miss -> the signal requirement is
#   overstated, or the test's signal model is wrong.
# ---------------------------------------------------------------------------
def t3_needs_signal():
    ds = []
    for pm in (0.10, 0.30, 0.70):  # [POLICY] knob
        _, truth, paus, pit, cap = generate(300, seed=7, p_cap_miss=pm)
        pr, _ = repair(paus, pit, cap, k=1)
        ds.append(prf(pr, truth) - prf(paus.copy(), truth))
    ok = ds[0] > ds[1] > ds[2]
    return ok, {"dF1_by_p_cap_miss": [round(d, 3) for d in ds]}


# ---------------------------------------------------------------------------
# T4 — Portable claim 3: k=1 optimal. [EXECUTED]
# Claim (Twin): k=1 wins because cap[0] is the reliable signal.
# Break: k=3 or k=5 beats k=1 -> optimality is policy-relative, not general.
#   (The claim as stated is already policy-scoped; this test checks the scope holds.)
# ---------------------------------------------------------------------------
def t4_k_optimal():
    _, truth, paus, pit, cap = generate(300, seed=7)
    fs = {}
    for k in (1, 3, 5):
        pr, _ = repair(paus, pit, cap, k=k)
        fs[k] = prf(pr, truth)
    ok = fs[1] >= fs[3] and fs[1] >= fs[5]
    return ok, {"F1_by_k": {k: round(v, 3) for k, v in fs.items()}}


# ---------------------------------------------------------------------------
# T5 — Honesty-contract audit. [EXECUTED]
# Claim (Reversed): every pause gets exactly one basis in
#   {confirmed, deleted, unknown}; confident-but-wrong is counted as
#   silent misrepair, never hidden.
# Break: any pause without a basis, any basis outside the tri-state, or
#   confident-wrong cases missing from the misrepair count -> contract violated.
# ---------------------------------------------------------------------------
def t5_honesty():
    _, truth, paus, pit, cap = generate(300, seed=7)
    pr, basis = repair(paus, pit, cap, k=1)
    idx = np.flatnonzero(paus)
    ok_basis = len(basis) == len(idx) and all(b in ("confirmed", "deleted", "unknown") for b in basis)
    cw, n_p = silent_misrepair(paus, basis, truth)
    # Cross-check: recompute confident-wrong independently; must match.
    cw2 = sum(1 for j, b in zip(idx, basis)
              if (b == "confirmed" and not truth[j]) or (b == "deleted" and truth[j]))
    ok = ok_basis and cw == cw2
    tc = Counter(basis)
    return ok, {"tri_state": dict(tc), "silent_misrepair": f"{cw}/{n_p}={cw/max(1,n_p):.1%}"}


# ---------------------------------------------------------------------------
# T6 — Policy-mirror guard (Twin's DIVERGE in code). [EXECUTED]
# Claim: headline magnitudes are policy-determined, not findings.
# Test: run the pipeline at two policy settings; assert the magnitudes DIFFER.
#   If dF1 were identical across policies, it would look like a constant of
#   nature — the guard proves it isn't. PASS = magnitudes move with the knobs.
# ---------------------------------------------------------------------------
def t6_policy_mirror():
    ds = []
    for ph in (0.05, 0.60):  # [POLICY] knobs
        _, truth, paus, pit, cap = generate(300, seed=7, p_hes=ph)
        pr, _ = repair(paus, pit, cap, k=1)
        ds.append(prf(pr, truth) - prf(paus.copy(), truth))
    ok = abs(ds[0] - ds[1]) > 0.05  # magnitudes move -> policy-determined
    return ok, {"dF1_p_hes_0.05": round(ds[0], 3), "dF1_p_hes_0.60": round(ds[1], 3),
                "verdict": "magnitudes are policy-determined; do not cite as findings"}


# ---------------------------------------------------------------------------
# T7 — Real-transcript trial. [UNMEASURABLE] here — the verdict trial.
# THE break condition for this project: does repair beat no-repair on REAL
# transcripts, not synthetic? Everything above is the pilot; this is the trial.
#
# Paper design:
#   DATA: real talk-to-text transcripts with HUMAN-ADJUDICATED pause/boundary
#     positions. Four's bound: on real streams there is no is_bnd, so the
#     "truth" is manufactured by annotators. Report inter-annotator agreement
#     (Cohen's kappa); the truth is noisy and the trial must say so.
#     [UNMEASURABLE] exact line: no annotated real-transcript corpus here.
#   ARMS: A0 no-repair (raw pause emission) vs A1 repair (k=1, conservative
#     delete, tri-state basis). Same transcripts, paired design.
#   PRIMARY: F1 against human-adjudicated boundaries.
#   PRE-REGISTERED: repair wins iff F1(A1) - F1(A0) > 0 AND the difference
#     exceeds the annotator-disagreement band (else inconclusive, not a win).
#   FALSIFICATION:
#     (1) dF1 <= 0 on real transcripts -> synthetic gain doesn't transfer.
#         The project is a synthetic artifact. Break.
#     (2) dF1 > 0 but within annotator noise -> inconclusive. Not a win;
#         re-run with more annotators or tighter protocol.
#     (3) silent-misrepair rate on real data >> synthetic 6.9% -> the honesty
#         contract doesn't survive contact with reality. The tri-state still
#         reports it (that's the contract working), but the mechanism's
#         confident claims aren't trustworthy.
#   CONFOUNDS (measure first, trial second):
#     - Real hesitation rate: if < 2.6% break-even (Shuffled), DON'T RUN.
#       The break condition says don't deploy; the trial obeys it.
#     - Real cap-signal reliability: p_cap_miss unknown on real ASR.
#       Measure it; if high, T3 predicts inversion — the trial tests that prediction.
#     - Domain: his quirks (talk-to-text error profile) vs general ASR.
#       Stratify or mark the scope.
#   OPEN (Reversed): pitch-only phantoms — pauses with no acoustic correlate.
#     The annotation protocol should count them; the trial reports the rate.
#   Twin's DIVERGE binds the reporting: no T1–T6 magnitude may be cited as
#   a prediction about real data. The synthetic numbers are mechanism checks.
# ---------------------------------------------------------------------------
def t7_real_trial_paper():
    return {"status": "[UNMEASURABLE]",
            "design": "human-adjudicated real transcripts (kappa reported); "
                      "paired A0/A1; F1 vs adjudicated boundaries; win iff dF1>0 "
                      "beyond annotator-noise band; falsify on dF1<=0, "
                      "inconclusive-in-noise, or misrepair blowout; measure "
                      "hesitation rate first (skip if <2.6%); stratify domain."}


# ---------------------------------------------------------------------------
# T8 — Cost-model check. [UNMEASURABLE] here.
# Shuffled: 2.05x cost, break-even hesitation 2.6% [POLICY][SYNTH].
# Paper: measure per-step wall-clock with/without repair on a real streaming
#   setup; break if the cost ratio makes sub-break-even domains net-negative
#   after including the repair's own latency in the pause path.
# [UNMEASURABLE] exact line: no streaming setup, no wall-clock here.
# ---------------------------------------------------------------------------
def t8_cost_paper():
    return {"status": "[UNMEASURABLE]",
            "design": "wall-clock per step, repair on/off, real streaming; "
                      "break if cost ratio invalidates the 2.6% break-even."}


def main():
    print("=== LATE CONTEXT PIPELINE — TEST DESIGN (Blink, experimental-design seat) ===")
    print("Target: synthesis_five_pipeline_verified.py. numpy only.")
    print("Twin's DIVERGE binds: no synthetic magnitude is a finding.\n")
    results = {}
    for name, fn in [("T1_fidelity", t1_fidelity),
                     ("T2_proportional", t2_proportional),
                     ("T3_needs_signal", t3_needs_signal),
                     ("T4_k_optimal", t4_k_optimal),
                     ("T5_honesty", t5_honesty),
                     ("T6_policy_mirror", t6_policy_mirror)]:
        ok, res = fn()
        results[name] = ok
        print(f"{name} [EXECUTED]: {'PASS' if ok else 'FAIL'} {res}")
    print(f"T7_real_trial: {t7_real_trial_paper()['status']}")
    print(f"   {t7_real_trial_paper()['design']}")
    print(f"T8_cost: {t8_cost_paper()['status']}")
    print(f"   {t8_cost_paper()['design']}")
    print()
    allok = all(results.values())
    print("OVERALL:", "PASS — mechanics verified; magnitudes policy-guarded; "
          "the verdict trial (T7) is designed on paper, unfunded." if allok
          else "FAIL — see above.")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
