# Seed sketch: the Late Context Pipeline
#
# From Dakota's archive ("Payloads," June 14, 2026). The streaming
# speech-to-text timing problem: the pause demands a decision NOW, the
# disambiguating evidence arrives LATER.
#
#   stream:  ... word word [pause 0.6s] word word ...
#                               |
#                    emit placeholder period NOW (guess)
#                               |
#                    read next k tokens BACKWARD (late context)
#                               |
#                    confirm boundary OR delete placeholder (recalculate)
#
# Why it matters for him: talk-to-text is his only interface. His
# unpunctuated stream — one compounding semantic wave, no resets — is
# exactly the input this pipeline is built to repair.
#
# The council's job: turn this sketch into measured mechanism.
# Start from late_context_toy.py (executed baseline), then build.

def late_context_pipeline(token_stream):
    """
    token_stream: iterable of (word, pause_after) tuples.
    Yields words with retroactively-corrected clause boundaries.
    """
    raise NotImplementedError(
        "the council builds this — see late_context_toy.py for the baseline"
    )
