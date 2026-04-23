"""
Rock Paper Scissors: Two-player game with hidden simultaneous moves.

Game logic in RPSTask(TaskSet), agents are separate, MultiAgentEnv runs the loop.
TaskSet controls information flow (build_prompt hides opponent's choice).
"""

import random

from datasets import Dataset

from verifiers.envs.agent import Agent
from verifiers.envs.multiagent_env import MultiAgentEnv
from verifiers.rubrics.multiagent_rubric import MultiAgentRubric
from verifiers.envs.taskset import TaskSet
from verifiers.types import Messages, State, SystemMessage, UserMessage


BEATS = {"rock": "scissors", "paper": "rock", "scissors": "paper"}


# =============================================================================
# Rubric
# =============================================================================

def create_rubric() -> MultiAgentRubric:
    rubric = MultiAgentRubric()

    def player1_reward(state, **kwargs) -> float:
        extras = state.get("extras", {})
        total = extras.get("round", 1)
        return extras.get("p1_score", 0) / total if total > 0 else 0.0

    def player2_reward(state, **kwargs) -> float:
        extras = state.get("extras", {})
        total = extras.get("round", 1)
        return extras.get("p2_score", 0) / total if total > 0 else 0.0

    rubric.add_actor_reward_func("player1", player1_reward, weight=1.0)
    rubric.add_actor_reward_func("player2", player2_reward, weight=1.0)
    return rubric


# =============================================================================
# TaskSet: ALL game logic lives here
# =============================================================================

class RPSTask(TaskSet):
    """
    Rock Paper Scissors — best of N rounds.

    Simultaneous moves via information hiding:
    player1 goes first, player2 goes second, but build_prompt
    hides player1's current-round choice from player2.
    """

    def __init__(self, num_rounds: int = 3, num_examples: int = -1):
        dataset = self._create_dataset()
        if num_examples > 0:
            dataset = dataset.select(range(min(num_examples, len(dataset))))

        super().__init__(
            name="rock_paper_scissors",
            dataset=dataset,
            rubric=create_rubric(),
            roles=["player1", "player2"],
        )
        self.num_rounds = num_rounds

    @staticmethod
    def _create_dataset() -> Dataset:
        return Dataset.from_list([
            {
                "prompt": [{"role": "user", "content": "play"}],
                "answer": "",
                "info": {"seed": i},
                "example_id": i,
                "task": "rock_paper_scissors",
            }
            for i in range(10)
        ])

    # ---- State ----

    async def setup_state(self, state: State) -> State:
        state["extras"]["round"] = 0
        state["extras"]["p1_score"] = 0
        state["extras"]["p2_score"] = 0
        state["extras"]["history"] = []        # [(p1_choice, p2_choice, result), ...]
        state["extras"]["p1_pending"] = None   # player1's choice waiting for player2
        return state

    # ---- Prompts ----

    async def build_prompt(self, role: str, state: State) -> Messages:
        round_num = state["extras"]["round"] + 1
        history = state["extras"]["history"]

        if role == "player1":
            strategy_hint = "You like to play aggressively and switch moves often to keep your opponent guessing."
        else:
            strategy_hint = "You like to play defensively and try to predict what your opponent will do next."

        system = (
            f"You are {role} in Rock Paper Scissors. Best of {self.num_rounds} rounds.\n"
            f"{strategy_hint}\n\n"
            "Respond with EXACTLY one word: rock, paper, or scissors\n"
            "Nothing else. Just the word."
        )

        if not history:
            history_str = "No rounds played yet."
        else:
            lines = []
            for i, (p1, p2, result) in enumerate(history, 1):
                you = p1 if role == "player1" else p2
                opp = p2 if role == "player1" else p1
                lines.append(f"Round {i}: You={you}, Opponent={opp} → {result}")
            history_str = "\n".join(lines)

        return [
            SystemMessage(content=system),
            UserMessage(content=f"Round {round_num} of {self.num_rounds}.\n\nHistory:\n{history_str}\n\nMake your choice:"),
        ]

    # ---- Game Logic ----

    async def on_turn_complete(self, state: State) -> None:
        if not state["trajectory"]:
            return

        last_step = state["trajectory"][-1]
        actor_id = last_step["extras"].get("actor_id", "")
        choice = self._extract_choice(last_step)

        if actor_id == "player1":
            state["extras"]["p1_pending"] = choice

        elif actor_id == "player2":
            p1_choice = state["extras"]["p1_pending"]
            p2_choice = choice
            state["extras"]["p1_pending"] = None

            if p1_choice == p2_choice:
                result = "draw"
            elif BEATS.get(p1_choice) == p2_choice:
                result = "player1 wins"
                state["extras"]["p1_score"] += 1
            else:
                result = "player2 wins"
                state["extras"]["p2_score"] += 1

            state["extras"]["history"].append((p1_choice, p2_choice, result))
            state["extras"]["round"] += 1
            print(f"[rps] round {state['extras']['round']}: p1={p1_choice}, p2={p2_choice} -> {result}")

    async def should_stop(self, state: State) -> bool:
        return state["extras"].get("round", 0) >= self.num_rounds

    async def on_game_end(self, state: State) -> None:
        p1 = state["extras"]["p1_score"]
        p2 = state["extras"]["p2_score"]
        if p1 > p2:
            state["extras"]["winner"] = "player1"
        elif p2 > p1:
            state["extras"]["winner"] = "player2"
        else:
            state["extras"]["winner"] = "tie"
        print(f"[rps] game over: p1={p1}, p2={p2}, winner={state['extras']['winner']}")

    # ---- Helpers ----

    def _extract_choice(self, step) -> str:
        completion = step.get("completion", [])
        if not completion:
            return "rock"
        text = completion[-1].get("content", "").lower().strip()
        for c in ["rock", "paper", "scissors"]:
            if c in text:
                return c
        return random.choice(["rock", "paper", "scissors"])


# =============================================================================
# Environment Loader
# =============================================================================

def load_environment(
    num_rounds: int = 3,
    num_examples: int = -1,
    p1_model: str | None = None,
    p2_model: str | None = None,
):
    """
    Composition:
        Task   = RPSTask (rules + prompts + scoring)
        Agents = {player1, player2}
        Env    = MultiAgentEnv(task, agents)

    Pass p1_model/p2_model to use different models per player.
    None = use the default model from the eval/training command.
    """
    task = RPSTask(num_rounds=num_rounds, num_examples=num_examples)

    player1 = Agent(
        id="player1",
        max_tokens=20,
        is_trainable=True,
        model=p1_model,
    )

    player2 = Agent(
        id="player2",
        max_tokens=20,
        is_trainable=True,
        model=p2_model,
    )

    return MultiAgentEnv(
        task=task,
        agents={"player1": player1, "player2": player2},
        max_turns=num_rounds * 2 + 2,
    )
