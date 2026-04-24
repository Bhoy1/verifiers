"""
Multi-agent rubric with per-actor rewards and advantages.

Extends Rubric with:
- Per-actor reward functions (different rewards for different actors)
- Per-actor GRPO advantages (within-actor normalization)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict

import verifiers as vf
from verifiers.rubrics.rubric import Rubric
from verifiers.types import RewardFunc, State


class MultiAgentRubric(Rubric):
    """
    Rubric with per-actor rewards and GRPO advantages.

    Advantages are computed within actor groups (e.g. solver vs solver),
    not across all actors, preventing unfair cross-actor comparisons.

    Usage:
        rubric = MultiAgentRubric()
        rubric.add_actor_reward_func("solver", my_solver_reward)
        rubric.add_actor_reward_func("generator", my_generator_reward)
    """

    def __init__(self, parser: vf.Parser | None = None):
        super().__init__(parser=parser)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Per-actor reward functions: actor_id -> [(func, weight), ...]
        self.actor_reward_funcs: dict[str, list[tuple[RewardFunc, float]]] = defaultdict(list)

        # Actor metadata: actor_id -> {"is_trainable": bool}
        self.actors: dict[str, dict] = {}

        # Register a group-level scoring marker so prime-rl uses run_group (not run_rollout).
        # The actual scoring is done by our score_group override, not this function.
        async def _multiagent_group_score(states: list, **kwargs) -> list:
            return [0.0] * len(states)
        self.add_reward_func(_multiagent_group_score, weight=0.0)

    def add_actor_reward_func(
        self,
        actor_id: str,
        func: RewardFunc,
        weight: float = 1.0,
    ) -> None:
        """Add a reward function specific to an actor."""
        self.actor_reward_funcs[actor_id].append((func, weight))

    def add_actor_metric(
        self,
        actor_id: str,
        func: RewardFunc,
    ) -> None:
        """Add a metric (zero-weight reward) for logging without affecting reward."""
        self.add_actor_reward_func(actor_id, func, weight=0.0)

    def register_actor(self, actor_id: str, is_trainable: bool = True) -> None:
        """Register an actor with trainability metadata.

        Called automatically by MultiAgentEnv during setup.
        """
        self.actors[actor_id] = {"is_trainable": is_trainable}

    def get_actor_id_from_state(self, state: State) -> str | None:
        """Extract actor ID from state extras."""
        return state.get("extras", {}).get("current_actor_id")

    async def _compute_actor_reward(
        self,
        state: State,
        actor_id: str,
    ) -> tuple[float, dict[str, float]]:
        """Compute reward for a single state using its actor's reward functions."""
        total_reward = 0.0
        metrics: dict[str, float] = {}

        actor_funcs = self.actor_reward_funcs.get(actor_id, [])
        for func, weight in actor_funcs:
            score = await self._call_individual_reward_func(func, state)
            score = score if score is not None else 0.0
            metrics[func.__name__] = score
            total_reward += score * weight

        return total_reward, metrics

    async def score_group(self, states: list[State]) -> None:
        """
        Score with per-actor GRPO advantages.

        1. Compute rewards per state using actor-specific reward functions
        2. Group states by actor_id
        3. Compute advantages within each actor group (actor vs same-actor peers)
        """
        if not states:
            self.logger.warning("No states to score")
            return

        start_time = time.time()

        # Score all states (compute rewards)
        await self._score_states(states)

        # Compute GRPO advantages per-actor group
        actor_groups: dict[str, list[State]] = defaultdict(list)
        for state in states:
            actor_id = self.get_actor_id_from_state(state) or "default"
            actor_groups[actor_id].append(state)

        for actor_id, actor_states in actor_groups.items():
            is_trainable = actor_states[0].get("is_trainable", True)
            if not is_trainable:
                for state in actor_states:
                    state["advantage"] = 0.0
                    for step in state.get("trajectory", []):
                        if step.get("advantage") is None:
                            step["advantage"] = 0.0
                        if step.get("reward") is None:
                            step["reward"] = state["reward"]
                continue

            actor_rewards = [s["reward"] for s in actor_states]
            mean_reward = sum(actor_rewards) / len(actor_rewards)
            print(
                f"[multiagent_rubric] actor={actor_id}, n={len(actor_states)}, "
                f"mean_reward={mean_reward:.4f}, rewards={[round(r, 3) for r in actor_rewards]}"
            )

            for state in actor_states:
                advantage = state["reward"] - mean_reward
                state["advantage"] = advantage
                for step in state.get("trajectory", []):
                    if step.get("advantage") is None:
                        step["advantage"] = advantage
                    if step.get("reward") is None:
                        step["reward"] = state["reward"]

        scoring_ms = (time.time() - start_time) * 1000
        for state in states:
            if "timing" in state:
                state["timing"]["scoring_ms"] = scoring_ms
                state["timing"]["total_ms"] += scoring_ms

    async def _score_states(self, states: list[State]) -> None:
        """Score a list of states with their actor-specific reward functions."""
        if not states:
            return

        actor_ids = [self.get_actor_id_from_state(s) or "default" for s in states]

        results = await asyncio.gather(*[
            self._compute_actor_reward(state, actor_id)
            for state, actor_id in zip(states, actor_ids)
        ])

        # Prefix metric names with actor_id so per-actor metrics show up in wandb
        # e.g. codegen_v1b/codegen_reward, codegen_v4/code_compiles_metric
        all_keys: set[str] = set()
        prefixed_results = []
        for (reward, metrics), actor_id in zip(results, actor_ids):
            prefixed = {f"{actor_id}/{k}": v for k, v in metrics.items()}
            prefixed_results.append((reward, prefixed))
            all_keys.update(prefixed.keys())

        for state, (reward, prefixed_metrics) in zip(states, prefixed_results):
            state["reward"] = reward
            state["metrics"] = {k: prefixed_metrics.get(k, 0.0) for k in all_keys}
