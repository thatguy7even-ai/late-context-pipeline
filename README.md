# Late Context Pipeline

A council code-dialogue project. From Dakota's archive (self-sent email,
"Payloads," June 14, 2026): the three-step late-context pipeline for
streaming speech-to-text — the mechanism behind his talk-to-text interface,
rebuilt as executable code.

## The idea

Streaming speech-to-text faces a timing problem: when the speaker pauses,
the engine must decide *now* whether a clause ended, but the evidence that
disambiguates arrives *later*. The pipeline:

1. **Local Decoding** — on pause/pitch-drop, emit a placeholder period
   immediately (a statistical guess, not a commitment).
2. **Late Context** — pull the next k tokens *backward* into the decision.
3. **Recalculation** — given the lookahead, confirm the boundary or delete
   the placeholder and reweight.

The archive's claim: unpunctuated talk-to-text forces the model to process
a compounding semantic wave with no attention-head resets; retroactive
correction restores the clause structure the stream never marked.

## What's here

- `seed_sketch.py` — the idea as a sketch (the starting point).
- `late_context_toy.py` — mother's toy, executed. Synthetic token streams
  with pause markers and known ground truth; naive placeholder thresholding
  vs lookahead correction, measured as precision/recall/F1 over 300 streams.
  Result: F1 0.605 → 0.625 (+0.020, precision-only gain). The mechanism is
  verified; the magnitude on real dictation is unmeasured.

## The test that matters

Run the pipeline on Dakota's actual dictation stream — real pauses, real
repairs — and measure whether retroactive correction beats naive
thresholding where it counts: his stream, not a synthetic grammar.

## Status

Seed stage. The council develops it through code dialogue: code alterations
passed seat to seat, each turn carrying the altered code, why it was written
that way, what it means, and what assumption moved. Executable claims get
executed; the rest is marked at the exact line.
