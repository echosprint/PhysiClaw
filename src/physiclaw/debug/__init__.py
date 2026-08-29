"""The e2e debug harness — a virtual user channel, off in production.

Three fakes and nothing else: the debug wake (`agent/hooks/debug.py`),
the user's chat bubble, and their confirm/deny replies. Everything
else runs real — the engine loop, the conductor's walk, the page
matcher, the reply tiers, the micro-calls, and the phone itself: a
gate's `channel/send` genuinely drives the rig and the ask really
lands in the IM thread.

The seam is the engine's dispatch, as a result TRANSFORMER: every call
executes for real, and only the OBSERVATION of conductor-minted
``channel/*`` macros and thread peeks is rewritten from
``debug/thread.json`` (`interceptor.py`), with listings rendered
against the real channel pack's own fingerprint (`thread.py`) so
`match_screen` and `reply.new_incoming` genuinely run.
`physiclaw debug --task --reply` scripts the whole conversation up
front: replies are STAGED, and `thread.py`'s timing rule releases each
one only on a peek that follows an agent ask — after the ask, never
into its baseline. The engine reaches this package only through an
env-gated dotted path (`agent.engine.plugins.load_debug_intercept`),
never an import — the same blindness rule as the conductor's plugin
seam.
"""
