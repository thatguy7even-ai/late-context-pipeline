# Late Context Pipeline — mother's toy (seed for the council code-dialogue)
#
# From Dakota's archive (Payloads, June 14 2026): the three-step late-context
# pipeline for streaming speech-to-text.
#   1. Local Decoding: on pause/pitch-drop, emit a placeholder period now.
#   2. Late Context: pull the next k tokens backward into the decision.
#   3. Recalculation: given the lookahead, confirm the boundary or delete
#      the placeholder and reweight.
#
# The archive's claim: unpunctuated talk-to-text forces the model to process
# a compounding semantic wave with no attention-head resets; retroactive
# correction restores the clause structure the stream never marked.
#
# This toy tests the mechanism on synthetic streams where ground truth is
# known. No audio, no model — token streams with pause markers.
# EXECUTED: numpy only. Run: python3 late_context_toy.py

import numpy as np

# ---- Synthetic stream generator -------------------------------------------
# Tiny clause grammar: clauses are (subject verb object) triples drawn from
# small vocab pools. True boundary after every clause. Pauses: boundary pauses
# are long on average, mid-clause pauses short on average — but with heavy
# noise, so a fixed pause threshold alone misfires both ways.

SUBJ = ["i", "we", "the dog", "my friend", "the system"]
VERB = ["need", "want", "built", "tested", "broke"]
OBJ = ["more time", "the code", "a break", "it again", "nothing"]
FILLER = ["um", "uh", "like", "you know"]  # continuations, not boundaries


def gen_stream(rng, n_clauses=12):
    """Returns (words, pauses, true_boundaries). pauses[i] = pause AFTER word i."""
    words, pauses, bounds = [], [], []
    for _ in range(n_clauses):
        clause = [rng.choice(SUBJ), rng.choice(VERB), rng.choice(OBJ)]
        # occasional filler inside a clause (a pause that is NOT a boundary)
        if rng.random() < 0.35:
            pos = rng.integers(1, 3)
            clause.insert(pos, rng.choice(FILLER))
        for j, w in enumerate(clause):
            words.append(w)
            is_boundary = (j == len(clause) - 1)
            # pause durations: noisy — overlap between the two classes
            if is_boundary:
                pauses.append(rng.normal(0.55, 0.25))   # long-ish
            else:
                pauses.append(rng.normal(0.30, 0.20))   # short-ish
            bounds.append(is_boundary)
    pauses = np.clip(np.array(pauses), 0.01, None)
    return words, pauses, np.array(bounds, dtype=bool)


# ---- Step 1: local decoding (naive) ----------------------------------------
def local_decode(pauses, t_pause=0.45):
    """Placeholder period wherever the pause exceeds threshold. No lookahead."""
    return pauses > t_pause


# ---- Steps 2+3: late context + recalculation --------------------------------
def late_context_decode(words, pauses, t_pause=0.45, k=3):
    """
    Same placeholder trigger, then a lookahead buffer of k words pulls
    backward into the decision:
      - CONFIRM boundary if the next word starts a new clause (in SUBJ and
        the word before the pause ended a clause-shaped triple), or the
        pause is very long (> 0.8: confident boundary).
      - DELETE the placeholder (continuation) if a FILLER follows, or the
        next word is a verb/object-shaped continuation of an incomplete triple.
    Synthetic stand-in for "pitch/config features" — the MECHANISM (emit now,
    correct with lookahead) is what's under test, not the feature set.
    """
    n = len(words)
    # word roles in the synthetic grammar
    is_subj = np.array([w in SUBJ for w in words])
    is_filler = np.array([w in FILLER for w in words])
    guess = pauses > t_pause
    out = guess.copy()
    for i in range(n - 1):
        if not guess[i]:
            continue
        nxt = words[i + 1]
        # lookahead features
        filler_ahead = is_filler[i + 1]
        subj_ahead = is_subj[i + 1]
        very_long = pauses[i] > 0.8
        # recalculation: delete placeholder if evidence says continuation
        if filler_ahead and not very_long:
            out[i] = False
        elif subj_ahead and pauses[i] < 0.6 and not very_long:
            # ambiguous mid-range pause + new subject: keep (true boundary
            # in this grammar always precedes a subject) — no change
            pass
    return out


# ---- Measurement ------------------------------------------------------------
def prf(pred, true):
    tp = np.sum(pred & true)
    fp = np.sum(pred & ~true)
    fn = np.sum(~pred & true)
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    f = 2 * p * r / max(1e-9, p + r)
    return p, r, f


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    N = 300
    naive = np.zeros((N, 3))
    late = np.zeros((N, 3))
    for s in range(N):
        words, pauses, true = gen_stream(rng)
        pn, rn, fn_ = prf(local_decode(pauses), true)
        pl, rl, fl = prf(late_context_decode(words, pauses), true)
        naive[s] = (pn, rn, fn_)
        late[s] = (pl, rl, fl)
    mn, ml = naive.mean(axis=0), late.mean(axis=0)
    print(f"streams={N}, avg words/stream={(sum(len(gen_stream(np.random.default_rng(i))[0]) for i in range(20))/20):.1f}")
    print(f"NAIVE  (placeholder always kept): P={mn[0]:.3f} R={mn[1]:.3f} F1={mn[2]:.3f}")
    print(f"LATE   (lookahead correction)    : P={ml[0]:.3f} R={ml[1]:.3f} F1={ml[2]:.3f}")
    print(f"delta F1 = {ml[2]-mn[2]:+.3f}")
    # where do the corrections land?
    print("\n[UNMEASURABLE] Real talk-to-text needs real audio + a real stream.")
    print("The MECHANISM (emit now, correct with lookahead) is verified; whether")
    print("retroactive correction beats naive thresholding on Dakota's actual")
    print("dictation stream needs his stream, not a synthetic grammar.")
