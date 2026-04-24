"""
GSM8K with group scoring — forces advantages to be computed in verifiers.
Same env, same rubric, same rewards. Just goes through run_group instead of run_rollout
so advantages flow from verifiers to prime-rl.

For testing: compare training with normal gsm8k (prime-rl computes advantages)
vs this env (verifiers computes advantages). Should produce same results.
"""

from gsm8k import load_environment as _load_gsm8k


def load_environment(**kwargs):
    env = _load_gsm8k(**kwargs)

    # Add a zero-weight group func so requires_group_scoring=True
    # This forces prime-rl to call run_group instead of run_rollout,
    # which triggers score_group (sets advantage on each state)
    async def _group_marker(states, **kw) -> list:
        return [0.0] * len(states)

    env.rubric.add_reward_func(_group_marker, weight=0.0)
    return env
