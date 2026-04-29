"""
LongMemEval Explore Environment

REPL-based exploration: model navigates conversation history via Python code,
manages its own notes, and answers questions.

The model has:
  - `history`: list of conversation sessions (stored in REPL, not in context window)
  - `question`: the question to answer
  - `context_list`: a list for the model's notes (persistent across turns)
  - `FINAL(answer)`: function to submit final answer

Each turn the model writes Python code that executes in a persistent namespace.
The model sees only the stdout/result — the history stays in the REPL.
"""

import asyncio
import io
import json
import logging
import os
import re
import sys
import urllib.request
from pathlib import Path

from datasets import Dataset

import verifiers as vf
from verifiers.types import UserMessage, SystemMessage

logger = logging.getLogger(__name__)


# ============================================================
# Data download
# ============================================================

DATA_URL = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"
DATA_DIR = Path(os.environ.get("LONGMEMEVAL_DATA_DIR", Path.home() / ".cache" / "longmemeval"))


def _download_data(variant: str = "longmemeval_s_cleaned") -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{variant}.json"
    filepath = DATA_DIR / filename
    if not filepath.exists():
        url = f"{DATA_URL}/{filename}"
        logger.info(f"Downloading {url} to {filepath}...")
        urllib.request.urlretrieve(url, filepath)
    return filepath


# ============================================================
# LLM Judge (same as longmemeval_context)
# ============================================================

def _get_judge_prompt(question_type, question, answer, response):
    is_abstention = question_type.endswith("_abs") or "abstention" in question_type.lower()
    if is_abstention:
        return (
            "I will give you an unanswerable question, an explanation, and a response from a model. "
            "Please answer yes if the model correctly identifies the question as unanswerable.\n\n"
            f"Question: {question}\n\nExplanation: {answer}\n\nModel Response: {response}\n\n"
            "Does the model correctly identify the question as unanswerable? Answer yes or no only."
        )
    elif question_type == "temporal-reasoning":
        return (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Do not penalize off-by-one errors.\n\n"
            f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    elif question_type == "knowledge-update":
        return (
            "I will give you a question, a correct answer, and a response from a model. "
            "If the response contains some previous information along with an updated answer, "
            "it should be correct as long as the updated answer is required.\n\n"
            f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    elif question_type == "single-session-preference":
        return (
            "I will give you a question, a rubric, and a response from a model. "
            "The response is correct as long as it recalls and utilizes user's personal information correctly.\n\n"
            f"Question: {question}\n\nRubric: {answer}\n\nModel Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    else:
        return (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer.\n\n"
            f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )


def _judge_reward_sync(question, answer, response, question_type, judge_model):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    prompt = _get_judge_prompt(question_type, question, answer, response)
    try:
        result = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=10,
            temperature=0.0,
        )
        judge_response = result.choices[0].message.content.strip().lower()
        return 1.0 if "yes" in judge_response else 0.0
    except Exception as e:
        logger.error(f"Judge call failed: {e}")
        return 0.0


# ============================================================
# Safe Python execution (persistent namespace)
# ============================================================

class PythonREPL:
    """Simple persistent Python REPL with namespace isolation."""

    def __init__(self):
        self.namespace = {"__builtins__": __builtins__}
        self._final_answer = None

    def seed(self, history, question):
        """Seed the REPL with data."""
        self.namespace["history"] = history
        self.namespace["question"] = question
        self.namespace["context_list"] = []

        # FINAL function
        def final_fn(answer):
            self._final_answer = str(answer)
            print(f"FINAL ANSWER: {self._final_answer}")
        self.namespace["FINAL"] = final_fn

    def execute(self, code: str, timeout: int = 30) -> str:
        """Execute code and return stdout + result."""
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        try:
            # Try exec first (statements)
            exec(code, self.namespace)
            output = buffer.getvalue()
            return output if output else "(no output)"
        except SyntaxError:
            # Try eval for expressions
            try:
                buffer = io.StringIO()
                sys.stdout = buffer
                result = eval(code, self.namespace)
                output = buffer.getvalue()
                if result is not None:
                    output += repr(result)
                return output if output else "(no output)"
            except Exception as e:
                return f"Error: {e}"
        except Exception as e:
            output = buffer.getvalue()
            return f"{output}Error: {e}"
        finally:
            sys.stdout = old_stdout

    @property
    def final_answer(self):
        return self._final_answer

    @property
    def context_list(self):
        return self.namespace.get("context_list", [])


# ============================================================
# System prompt
# ============================================================

EXPLORE_PROMPT_VERSIONS = {
    "light": """You have a conversation history stored in the `history` variable (a list of session strings).
Your task is in the `question` variable.
You can use `context_list` to save notes as you explore.

Use the python tool to run code. Use print() to see data. Examples:
  print(len(history))
  print(history[5][:200])
  print(history[5])
  context_list.append("user graduated Business Admin")
  print(context_list)

IMPORTANT: You MUST submit your final answer by calling FINAL("your answer") in a python tool call.
Do NOT write your answer as plain text. Always use the FINAL() function in code.

Explore efficiently — you don't need to read every session.""",

    "notes": """You have a conversation history stored in the `history` variable (a list of session strings).
Your task is in the `question` variable.

You MUST use `context_list` to build up evidence as you explore. Save every relevant fact you find before answering.

Strategy:
1. Scan history to understand its size and structure
2. Search for relevant sessions using keywords
3. For each relevant session, extract key facts and append them to context_list
4. Review your collected notes in context_list
5. Call FINAL("your answer") based on your notes

Use the python tool to run code. Use print() to see data. Examples:
  print(len(history))
  for i,s in enumerate(history):
      if 'keyword' in s.lower():
          print(i, s[:300])
  context_list.append("Session 12: user graduated Business Admin in 2019")
  context_list.append("Session 34: user mentioned MBA plans")
  print(context_list)
  FINAL("Business Administration")

IMPORTANT: You MUST submit your final answer by calling FINAL("your answer") in a python tool call.
Do NOT write your answer as plain text. Always use the FINAL() function in code.

Do NOT call FINAL() until you have gathered evidence in context_list.""",
}

EXPLORE_SYSTEM_PROMPT = EXPLORE_PROMPT_VERSIONS["light"]


# ============================================================
# Explore environment
# ============================================================

class LongMemEvalExploreEnv(vf.StatefulToolEnv):
    """
    REPL-based exploration of LongMemEval conversation history.

    Model uses a python tool to explore history stored in a persistent namespace.
    Model manages context_list (notes) and calls FINAL() when ready.
    """

    def __init__(self, judge_model: str = "gpt-5.4", max_turns: int = 30, **kwargs):
        self.judge_model = judge_model
        self._current_repl_result = None
        # Define the python tool
        tools = [self._python_tool]
        super().__init__(tools=tools, max_turns=max_turns, **kwargs)

    @staticmethod
    async def _python_tool(code: str) -> str:
        """Execute Python code in the persistent REPL environment.

        Args:
            code: Python code to execute. Use print() to see output.

        Returns:
            The stdout output from executing the code.
        """
        # This return value is overridden — see update_tool_args
        return code

    def update_tool_args(self, tool_name, tool_args, messages, state, **kwargs):
        if tool_name == "_python_tool":
            repl = state["_repl"]
            code = tool_args.get("code", "")
            result = repl.execute(code)

            # Truncate if too long
            if len(result) > 4000:
                result = result[:4000] + "\n... (truncated)"

            self._current_repl_result = result
            state["_last_repl_result"] = result
            state["_turns_used"] = state.get("_turns_used", 0) + 1

            logger.info(f"[REPL] code={code[:100]}... result={result[:100]}...")

            if repl.final_answer is not None:
                state["_final_answer"] = repl.final_answer
        return tool_args

    async def call_tool(self, tool_name, tool_args, tool_call_id, **kwargs):
        """Override to return REPL result instead of calling the placeholder function."""
        if tool_name == "_python_tool":
            result = self._current_repl_result or "(no output)"
            return vf.ToolMessage(role="tool", content=result, tool_call_id=tool_call_id)
        return await super().call_tool(tool_name, tool_args, tool_call_id, **kwargs)

    async def setup_state(self, state):
        state = await super().setup_state(state)
        info = state.get("info", {})

        # Create REPL and seed with data
        sessions = json.loads(info["sessions_json"])
        question = info["question"]

        repl = PythonREPL()
        repl.seed(sessions, question)
        state["_repl"] = repl
        state["_final_answer"] = None
        state["_turns_used"] = 0

        logger.info(f"[Explore] Seeded REPL: {len(sessions)} sessions, q={question[:50]}...")
        return state

    @vf.stop
    async def answer_submitted(self, state) -> bool:
        return state.get("_final_answer") is not None

    async def render_completion(self, state):
        # Use default render (includes tool messages from trajectory prompts)
        await super().render_completion(state)
        # Export state columns
        dict.__setitem__(state, "_final_answer", state.get("_final_answer", ""))
        dict.__setitem__(state, "_turns_used", state.get("_turns_used", 0))
        repl = state.get("_repl")
        dict.__setitem__(state, "_context_list", repl.context_list if repl else [])


# ============================================================
# Reward function
# ============================================================

class ExploreRubric(vf.Rubric):
    def __init__(self, judge_model="gpt-5.4", max_turns=30):
        super().__init__()
        self.judge_model = judge_model
        self.max_turns = max_turns
        self.add_reward_func(self.judge_with_efficiency, weight=1.0)

    async def judge_with_efficiency(self, state) -> float:
        info = state.get("info", {})
        answer = info.get("answer", "")
        question = info.get("question", "")
        question_type = info.get("question_type", "")

        final_answer = state.get("_final_answer", "")
        if not final_answer:
            return 0.0

        # Judge accuracy
        loop = asyncio.get_running_loop()
        accuracy = await loop.run_in_executor(
            None, _judge_reward_sync,
            question, str(answer), final_answer, question_type, self.judge_model,
        )

        # Efficiency bonus: fewer turns = better
        turns_used = state.get("_turns_used", self.max_turns)
        efficiency = max(0, 1 - turns_used / self.max_turns)
        reward = accuracy * (1.0 + 0.2 * efficiency)

        return reward


def turns_used_metric(state) -> float:
    return float(state.get("_turns_used", 0))


def has_answer_metric(state) -> float:
    return 1.0 if state.get("_final_answer") else 0.0


def context_list_size_metric(state) -> float:
    repl = state.get("_repl")
    if repl:
        return float(len(repl.context_list))
    return 0.0


# ============================================================
# Dataset builder
# ============================================================

def _build_dataset(variant, n, question_ids, prompt_version="light"):
    def builder():
        filepath = _download_data(variant)
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        if question_ids:
            qid_set = set(question_ids)
            data = [e for e in data if e["question_id"] in qid_set]
        if n > 0:
            data = data[:n]

        system_prompt = EXPLORE_PROMPT_VERSIONS.get(prompt_version, EXPLORE_PROMPT_VERSIONS["light"])

        rows = []
        for entry in data:
            sessions = entry["haystack_sessions"]
            dates = entry.get("haystack_dates", [""] * len(sessions))
            # Format sessions as readable strings
            formatted = []
            for sess, date in zip(sessions, dates):
                lines = []
                if date:
                    lines.append(f"[Date: {date}]")
                for turn in sess:
                    lines.append(f"{turn['role']}: {turn['content']}")
                formatted.append("\n".join(lines))

            rows.append({
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Your question is: {entry['question']}\n\nYou have {len(sessions)} sessions in `history`. Start exploring."},
                ],
                "answer": str(entry.get("answer", "")),
                "info": json.dumps({
                    "question_id": entry["question_id"],
                    "question": entry["question"],
                    "answer": str(entry.get("answer", "")),
                    "question_type": entry.get("question_type", ""),
                    "question_date": entry.get("question_date", ""),
                    "sessions_json": json.dumps(formatted),
                }),
            })
        from datasets import Value
        ds = Dataset.from_list(rows)
        # Cast info to large_string for very long sessions (longmemeval_m has ~5M chars per row)
        new_features = ds.features.copy()
        new_features["info"] = Value("large_string")
        ds = ds.cast(new_features)
        return ds
    return builder


# ============================================================
# Entry point
# ============================================================

def load_environment(
    variant: str = "longmemeval_s_cleaned",
    num_examples: int = -1,
    question_ids: str = "",
    judge_model: str = "gpt-5.4",
    max_turns: int = 30,
    prompt_version: str = "light",
) -> vf.Environment:
    """
    Load LongMemEval explore environment.

    Args:
        variant: Dataset variant
        num_examples: Limit examples (-1 for all)
        question_ids: Comma-separated question IDs to filter
        judge_model: Model for LLM judge
        max_turns: Max REPL turns before timeout (affects efficiency bonus)
    """
    qids = [q.strip() for q in question_ids.split(",") if q.strip()] if question_ids else None
    dataset_builder = _build_dataset(variant, num_examples, qids, prompt_version=prompt_version)
    rubric = ExploreRubric(judge_model=judge_model, max_turns=max_turns)

    env = LongMemEvalExploreEnv(
        dataset=dataset_builder,
        rubric=rubric,
        judge_model=judge_model,
        max_turns=max_turns,
    )
    env.rubric.add_metric(turns_used_metric, weight=0.0)
    env.rubric.add_metric(has_answer_metric, weight=0.0)
    env.rubric.add_metric(context_list_size_metric, weight=0.0)
    return env
