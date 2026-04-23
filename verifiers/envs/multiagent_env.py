"""
Multi-agent environment with turn-based actor management.

Extends Environment directly (not MultiTurnEnv) because multi-agent
rollouts need a fundamentally different loop — multiple actors per round,
each potentially with different models/clients.

Two modes:
    TaskSet mode:  MultiAgentEnv(task=my_task, agents={"a": agent_a, "b": agent_b})
    Subclass mode: class MyEnv(MultiAgentEnv) — override hooks directly
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from datasets import Dataset

import verifiers as vf
from verifiers.clients import Client, OpenAIChatCompletionsClient, resolve_client
from verifiers.types import (
    ClientConfig,
    Messages,
    RolloutInput,
    RolloutOutput,
    SamplingArgs,
    State,
    TrajectoryStep,
)
from verifiers.utils.async_utils import maybe_retry
from verifiers.utils.client_utils import resolve_client_config
from verifiers.utils.message_utils import maybe_normalize_messages
from verifiers.utils.response_utils import parse_response_message, parse_response_tokens
from verifiers.utils.save_utils import state_to_output

logger = logging.getLogger(__name__)


class MultiAgentEnv(vf.Environment):
    """
    Multi-agent environment.

    TaskSet mode:  MultiAgentEnv(task=my_task, agents={"a": agent_a})
    Subclass mode: class MyEnv(MultiAgentEnv) — set self.actors/self.name,
                   override hooks, agents injected by Registry.
    """

    def __init__(
        self,
        task: Any | None = None,
        agents: dict[str, Any] | list[Any] | None = None,
        max_turns: int = -1,
        parallel_actors: bool = False,
        **kwargs,
    ):
        self._task = task
        self.max_turns = max_turns
        self.parallel_actors = parallel_actors

        # Normalize agents to dict
        if agents is not None:
            if isinstance(agents, dict):
                self._agents = agents
            else:
                self._agents = {a.id: a for a in agents}
        else:
            self._agents = {}

        # TaskSet mode: wire dataset/rubric/roles
        if task is not None:
            self.actors = list(self._agents.keys())

            for role in task.roles:
                if role not in self._agents:
                    raise ValueError(
                        f"Task role '{role}' has no agent. "
                        f"Available agents: {list(self._agents.keys())}"
                    )

            if "dataset" not in kwargs and "eval_dataset" not in kwargs:
                kwargs["dataset"] = task.get_examples()
            if "rubric" not in kwargs and task.rubric is not None:
                kwargs["rubric"] = task.rubric
            self.name = task.name

        # Worker envs: dummy dataset to satisfy Environment base class
        if "dataset" not in kwargs and "eval_dataset" not in kwargs:
            kwargs["dataset"] = Dataset.from_dict({
                "prompt": [[{"role": "user", "content": ""}]],
                "answer": [""],
                "example_id": [0],
                "task": ["dummy"],
            })

        super().__init__(**kwargs)

        self._register_actors_with_rubric()

    # -------------------------------------------------------------------------
    # Actor / Agent Lookup
    # -------------------------------------------------------------------------

    def get_actor(self, actor_id: str) -> Any:
        """Get an agent by ID."""
        if actor_id in self._agents:
            return self._agents[actor_id]
        raise KeyError(
            f"Agent '{actor_id}' not found. "
            f"Available: {list(self._agents.keys())}"
        )

    def inject_agents(self, agents: dict[str, Any]) -> None:
        """Add agents from Registry. Doesn't overwrite existing."""
        for aid, agent in agents.items():
            if aid not in self._agents:
                self._agents[aid] = agent
        self._register_actors_with_rubric()

    def _register_actors_with_rubric(self) -> None:
        """Register all agents with MultiAgentRubric for per-actor scoring."""
        from verifiers.rubrics.multiagent_rubric import MultiAgentRubric

        if hasattr(self, "rubric") and isinstance(self.rubric, MultiAgentRubric):
            for agent_id, agent in self._agents.items():
                self.rubric.register_actor(
                    agent_id, is_trainable=agent.is_trainable
                )

    # -------------------------------------------------------------------------
    # Turn Management (delegates to TaskSet, or subclass overrides)
    # -------------------------------------------------------------------------

    def get_active_actors(self, state: State) -> list[str]:
        """Return list of actors that should act this turn."""
        if self._task is not None:
            return self._task.get_active_roles(state)
        current = state["extras"].get("current_actor_id")
        if current is None:
            return [self.get_initial_actor(state)]
        return [self.get_next_actor(state)]

    def get_initial_actor(self, state: State) -> str:
        if self._task is not None:
            return self._task.get_initial_role(state)
        raise NotImplementedError("Subclass must override get_initial_actor()")

    def get_next_actor(self, state: State) -> str:
        if self._task is not None:
            return self._task.get_next_role(state)
        raise NotImplementedError("Subclass must override get_next_actor()")

    # -------------------------------------------------------------------------
    # State Setup
    # -------------------------------------------------------------------------

    async def setup_state(self, state: State) -> State:
        state["child_states"] = []
        state["extras"] = {
            "current_actor_id": None,
            "actor_history": [],
            "episode_id": state.get("trajectory_id", uuid.uuid4().hex),
            "parent_episode_id": None,
        }

        if hasattr(self, "registry"):
            state["registry"] = self.registry

        if self._task is not None:
            state = await self._task.setup_state(state)
        return state

    # -------------------------------------------------------------------------
    # Game Hooks (delegates to TaskSet)
    # -------------------------------------------------------------------------

    async def build_actor_prompt(self, actor_id: str, state: State) -> Messages:
        if self._task is not None:
            return await self._task.build_prompt(actor_id, state)
        raise NotImplementedError("Subclass must override build_actor_prompt()")

    async def on_turn_complete(self, state: State) -> None:
        if self._task is not None:
            return await self._task.on_turn_complete(state)

    async def on_game_end(self, state: State) -> None:
        if self._task is not None:
            return await self._task.on_game_end(state)

    async def env_response(self, messages: Messages, state: State, **kwargs) -> Messages | str:
        return []

    # -------------------------------------------------------------------------
    # Stop Condition
    # -------------------------------------------------------------------------

    @vf.stop(priority=100)
    async def has_error(self, state: State) -> bool:
        return state.get("error") is not None

    @vf.stop
    async def prompt_too_long(self, state: State) -> bool:
        return state.get("prompt_too_long", False)

    @vf.stop
    async def max_turns_reached(self, state: State) -> bool:
        max_turns = getattr(self, "max_turns", -1)
        return len(state["trajectory"]) >= max_turns and max_turns > 0

    @vf.stop
    async def task_stopped(self, state: State) -> bool:
        if self._task is not None:
            return await self._task.should_stop(state)
        return False

    # -------------------------------------------------------------------------
    # Trajectory Management
    # -------------------------------------------------------------------------

    async def add_trajectory_step(
        self, state: State, trajectory_step: TrajectoryStep
    ) -> None:
        current_actor_id = state["extras"]["current_actor_id"]
        if current_actor_id:
            trajectory_step["extras"]["actor_id"] = current_actor_id
            turn_index = len(state["trajectory"])
            state["extras"]["actor_history"].append((current_actor_id, turn_index))
        state["trajectory"].append(trajectory_step)

    async def add_model_response(
        self,
        state: State,
        prompt_messages: Messages,
        response,
    ):
        completion_messages = await parse_response_message(response)
        tokens = await parse_response_tokens(response, self.max_seq_len)
        response_is_truncated = response.message.is_truncated or False
        is_truncated = response_is_truncated or (
            tokens is not None and bool(tokens.get("is_truncated"))
        )
        trajectory_step = TrajectoryStep(
            prompt=prompt_messages,
            completion=completion_messages,
            response=response,
            tokens=tokens,
            reward=None,
            advantage=None,
            is_truncated=is_truncated,
            trajectory_id=state["trajectory_id"],
            extras={},
        )
        await self.add_trajectory_step(state, trajectory_step)

    # -------------------------------------------------------------------------
    # Rollout Loop
    # -------------------------------------------------------------------------

    async def _prepare_and_call(self, actor_id: str, state: State, sampling_args):
        """Build prompt and get LLM response for one actor. Returns (actor_id, prompt, response)."""
        from verifiers.types import SystemMessage

        prompt_messages = await self.build_actor_prompt(actor_id, state)
        prompt_messages = maybe_normalize_messages(
            prompt_messages, field_name="prompt_messages"
        )

        actor = self.get_actor(actor_id)
        merged_args = actor.merge_sampling_args(sampling_args or {})

        if actor.system_prompt:
            prompt_messages = [
                SystemMessage(content=actor.system_prompt),
                *prompt_messages,
            ]

        actor_client = actor.client
        if actor_client is not None and not isinstance(actor_client, Client):
            actor_client = OpenAIChatCompletionsClient(actor_client)

        actor_models = getattr(self, "_actor_models", {})
        used_model = actor_models.get(actor_id) or actor.model or state.get("model", "default")

        response = await self.get_model_response(
            state,
            prompt_messages,
            client=actor_client,
            model=used_model,
            sampling_args=merged_args,
        )

        return actor_id, prompt_messages, response

    async def rollout(
        self,
        input: RolloutInput,
        client: Client,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        state = await self.init_state(input, client, model, sampling_args)
        try:
            state = await self.setup_state(state)
        except vf.Error as e:
            state["error"] = e
            return state

        while not await self.is_completed(state):
            active_actors = self.get_active_actors(state)

            if self.parallel_actors and len(active_actors) > 1:
                # Parallel: fire all LLM calls concurrently, then add responses sequentially
                try:
                    results = await asyncio.gather(*[
                        self._prepare_and_call(aid, state, sampling_args)
                        for aid in active_actors
                    ])
                    for actor_id, prompt_messages, response in results:
                        state["extras"]["current_actor_id"] = actor_id
                        await self.add_model_response(state, prompt_messages, response)
                        await self.on_turn_complete(state)
                except vf.OverlongPromptError:
                    state["prompt_too_long"] = True
                    state["is_truncated"] = True
                except vf.Error as e:
                    state["error"] = e
            else:
                # Sequential: one actor at a time (default)
                for actor_id in active_actors:
                    state["extras"]["current_actor_id"] = actor_id
                    try:
                        _, prompt_messages, response = await self._prepare_and_call(
                            actor_id, state, sampling_args
                        )
                        await self.add_model_response(state, prompt_messages, response)
                        await self.on_turn_complete(state)
                    except vf.OverlongPromptError:
                        state["prompt_too_long"] = True
                        state["is_truncated"] = True
                        break
                    except vf.Error as e:
                        state["error"] = e
                        break

            if await self.is_completed(state):
                break

        await self.on_game_end(state)
        await self.render_completion(state)
        return state

    async def render_completion(self, state: State):
        """Build completion from all actors' turns, tagged with actor IDs."""
        if not state.get("trajectory"):
            state["completion"] = []
            return
        all_messages = []
        for step in state["trajectory"]:
            actor_id = step["extras"].get("actor_id", "")
            for msg in step.get("completion", []):
                tagged = dict(msg)
                if actor_id and tagged.get("content"):
                    tagged["content"] = f"[{actor_id}] {tagged['content']}"
                all_messages.append(tagged)
        state["completion"] = all_messages

    # -------------------------------------------------------------------------
    # Per-Actor State Creation (for splitting a game into per-actor outputs)
    # -------------------------------------------------------------------------

    SHARED_STATE_FIELDS = {
        "client",
        "model",
        "trajectory_id",
        "sampling_args",
    }

    def create_actor_state(
        self,
        parent_state: State,
        actor_id: str,
        actor_trajectory: list[TrajectoryStep],
    ) -> State:
        """Create a State containing only one actor's trajectory steps."""
        actor_state = State()

        for key in parent_state.keys():
            if key in self.SHARED_STATE_FIELDS:
                actor_state[key] = parent_state[key]

        if "timing" in parent_state:
            actor_state["timing"] = dict(parent_state["timing"])

        actor_state["answer"] = parent_state.get("answer", "")
        actor_state["task"] = parent_state.get("task", "")
        actor_state["example_id"] = parent_state.get("example_id", 0)
        actor_state["info"] = parent_state.get("info", {})
        actor_state["trajectory"] = actor_trajectory
        actor_state["extras"] = {
            **parent_state.get("extras", {}),
            "current_actor_id": actor_id,
        }
        actor_state["child_states"] = []
        actor_state["reward"] = None
        actor_state["advantage"] = None
        actor_state["metrics"] = None

        actor = self.get_actor(actor_id)
        actor_state["is_trainable"] = actor.is_trainable

        if actor_trajectory:
            raw_prompt = actor_trajectory[0].get("prompt", [])
            prompt_ref = raw_prompt
            for i in range(len(raw_prompt) - 1, -1, -1):
                if raw_prompt[i].get("role") == "system":
                    prompt_ref = raw_prompt[i:]
                    break
            actor_state["prompt"] = prompt_ref

            all_completions = []
            for step in actor_trajectory:
                all_completions.extend(step.get("completion", []))
            actor_state["completion"] = all_completions
        else:
            actor_state["prompt"] = parent_state.get("prompt", [])
            actor_state["completion"] = []

        return actor_state

    def create_actor_states(self, state: State, actor_ids: list[str] | None = None) -> list[State]:
        """Split a game state into per-actor states (one per trainable actor)."""
        if actor_ids is None:
            actor_ids = self.actors

        actor_states = []
        for actor_id in actor_ids:
            actor_trajectory = [
                step for step in state.get("trajectory", [])
                if step.get("extras", {}).get("actor_id") == actor_id
            ]
            new_state = self.create_actor_state(state, actor_id, actor_trajectory)
            if not actor_trajectory:
                print(f"[create_actor_states] actor={actor_id} has no trajectory steps (no actions taken)")
            actor_states.append(new_state)

        return actor_states

    # -------------------------------------------------------------------------
    # Training Support — override run_group to split into per-actor outputs
    # -------------------------------------------------------------------------

    async def run_group(
        self,
        group_inputs: list[RolloutInput],
        client: Client | ClientConfig,
        model: str,
        sampling_args: SamplingArgs,
        max_retries: int = 0,
        state_columns: list[str] | None = None,
        env_client=None,
        actor_models: dict[str, str] | None = None,
        **kwargs,
    ) -> list[RolloutOutput]:
        """Run a group of rollouts, splitting into per-actor outputs.

        Each game produces one output per trainable actor. If prime-rl
        requests N rollouts and we have K trainable actors, we run N/K
        games and return N total outputs.
        """
        env_client = env_client or getattr(self, "env_client", None)
        if env_client is not None:
            resolved_config = (
                resolve_client_config(client)
                if isinstance(client, ClientConfig)
                else None
            )
            if resolved_config is None:
                raise ValueError(
                    f"client must be ClientConfig in server mode, got {type(client)}"
                )
            return await env_client.run_group(
                group_inputs,
                resolved_config,
                model,
                sampling_args,
                max_retries,
                state_columns,
            )

        resolved_client = resolve_client(client)
        state_columns = list(state_columns or [])

        trainable_ids = [
            aid for aid, a in self._agents.items() if a.is_trainable
        ]
        num_trainable = len(trainable_ids) or 1

        if len(group_inputs) % num_trainable != 0:
            raise ValueError(
                f"rollouts_per_example ({len(group_inputs)}) must be divisible by "
                f"num_trainable_actors ({num_trainable}). "
                f"Each game produces {num_trainable} training outputs."
            )

        games_count = len(group_inputs) // num_trainable
        game_inputs = group_inputs[:games_count]
        print(f"[run_group] {len(group_inputs)} inputs, {num_trainable} trainable actors -> {games_count} games")

        self._actor_models = actor_models or {}

        async def attempt() -> list[State]:
            game_states = await asyncio.gather(*[
                self.rollout(inp, resolved_client, model, sampling_args)
                for inp in game_inputs
            ])
            # Split each game into per-actor states
            actor_states = []
            for state in game_states:
                per_actor = self.create_actor_states(state, actor_ids=trainable_ids)
                for astate in per_actor:
                    aid = astate.get("extras", {}).get("current_actor_id", "?")
                    traj_len = len(astate.get("trajectory", []))
                    print(f"[create_actor_states] actor={aid}, traj_steps={traj_len}")
                actor_states.extend(per_actor)

            if self.score_rollouts:
                await self.rubric.score_group(actor_states)
            else:
                await self.rubric.dummy_score_group(actor_states)

            for state in actor_states:
                aid = state.get("extras", {}).get("current_actor_id", "?")
                reward = state.get("reward", 0.0)
                advantage = state.get("advantage")
                print(f"[scored] actor={aid}, reward={reward:.4f}, advantage={advantage}")
                await self.rubric.cleanup(state)

            return actor_states

        actor_states = await maybe_retry(attempt, max_retries=max_retries)()
        return [state_to_output(s, state_columns) for s in actor_states]

    # -------------------------------------------------------------------------
    # vf-eval support — TODO: wire up generate() once environments are ported
    # -------------------------------------------------------------------------

    def get_actor_id_from_state(self, state: State) -> str:
        """Extract actor_id from state for results/logging."""
        return state.get("extras", {}).get("current_actor_id", "unknown")
