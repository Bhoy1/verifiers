"""
LongMemEval Context Management Environment

Evaluates LLM long-term memory via the LongMemEval benchmark.
Two modes:
  - baseline: Full conversation history dumped per question (SingleTurnEnv)
  - context:  Incremental session-by-session fact extraction into a context list,
              then answer from compressed facts (MultiTurnEnv)

Unlike LoCoMo, each question has its own unique haystack of ~40 sessions.
No caching across questions — every rollout processes its own sessions.

Scoring: LLM judge (per-question-type prompts, yes/no).
"""

import asyncio
import json
import logging
import os
import re
import urllib.request
from collections import Counter
from pathlib import Path

from datasets import Dataset

import verifiers as vf
from verifiers.types import SystemMessage, UserMessage
from verifiers.utils.message_utils import concat_messages, maybe_normalize_messages

logger = logging.getLogger(__name__)


# ============================================================
# Tokenizer
# ============================================================

_enc = None
try:
    import tiktoken
    for _encoding_name in ("o200k_base", "cl100k_base"):
        try:
            _enc = tiktoken.get_encoding(_encoding_name)
            break
        except (KeyError, ValueError):
            continue
    if _enc is None:
        _enc = tiktoken.encoding_for_model("gpt-4")

    def _count_tokens(text: str) -> int:
        return len(_enc.encode(text or ""))

    def _tokenize(text: str) -> list:
        return _enc.encode(text or "")
except ImportError:
    def _count_tokens(text: str) -> int:
        return len(text or "") // 4

    def _tokenize(text: str) -> list:
        return list(text or "")


# ============================================================
# RestrictedPython
# ============================================================

try:
    from RestrictedPython import compile_restricted_exec, limited_builtins, utility_builtins
    from RestrictedPython.Guards import guarded_iter_unpack_sequence, safer_getattr
    from RestrictedPython.PrintCollector import PrintCollector
    _HAS_RESTRICTED_PYTHON = True
except ImportError:
    _HAS_RESTRICTED_PYTHON = False


# ============================================================
# System prompts (from run_context_generation.py)
# ============================================================

CONTEXT_SYSTEM_PROMPT = """You manage a persistent context list that stores key facts from an ongoing conversation between a user and an assistant.

Rules:
- Extract concise facts from the dialogue. Do NOT copy dialogue verbatim.
- Each fact should be its own entry. Do not combine multiple facts into one entry.
- ALWAYS save: dates, times, names, locations, specific details, preferences, sentiments, and attitudes.
- Append new facts as new entries. Update existing entries when information changes.
- When the context list gets long, compress or pop the least important entries to make room.

Output your edits inside <edit> tags. Available operations:
  context_list.append("new fact")
  context_list[i] = "updated fact"
  context_list.pop(i)

Example turn:
  Current context list: ["User likes hiking", "User works at Google"]
  New dialogue:
  user: I just got promoted to senior engineer!

  <edit>context_list[1] = "User is senior engineer at Google (recently promoted)"</edit>"""

BASELINE_SYSTEM_PROMPT = """You are a helpful assistant with access to conversation history. Answer questions about the conversations concisely and accurately."""

QA_SYSTEM_PROMPT = "Answer the question concisely using the provided facts."

COMPACTION_SYSTEM_PROMPT = """You are given a conversation history between a user and an assistant. Extract all important facts into a Python list called context_list. Save key details: names, dates, locations, preferences, decisions, specific values, sentiments, and attitudes.

Output your context_list inside <edit> tags:
<edit>
context_list.append("fact 1")
context_list.append("fact 2")
...
</edit>"""


# ============================================================
# Conversation formatting
# ============================================================

def _format_session(session: list[dict], date: str = "") -> str:
    lines = []
    if date:
        lines.append(f"[Date: {date}]")
    for turn in session:
        role = turn.get("role", "unknown")
        content = turn.get("content", "").strip()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _format_all_sessions(sessions: list[list[dict]], dates: list[str]) -> str:
    parts = []
    for i, (session, date) in enumerate(zip(sessions, dates)):
        parts.append(f"### Session {i + 1}:\n{_format_session(session, date)}")
    return "\n\n".join(parts)


# ============================================================
# Context list helpers
# ============================================================

def _serialize_context_list(context_list: list[str]) -> str:
    if not context_list:
        return "[]"
    items = ", ".join(f'"{item}"' for item in context_list)
    return f"[{items}]"


def _extract_edit_code(response: str) -> str | None:
    match = re.search(r"<edit>(.*?)</edit>", response, re.DOTALL)
    if not match:
        return None
    code = match.group(1).strip()
    # Strip "context_list = []" reassignment — model sometimes resets the list
    code = re.sub(r'^context_list\s*=\s*\[\s*\]\s*\n?', '', code, flags=re.MULTILINE)
    return code.strip() if code.strip() else None


def _extract_answer_tag(response: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    text = re.sub(r"<edit>.*?</edit>", "", response, flags=re.DOTALL).strip()
    return text


def _execute_context_edit(edit_code: str, context_list: list[str]) -> tuple[bool, str]:
    if not edit_code or not edit_code.strip():
        return True, ""
    if _HAS_RESTRICTED_PYTHON:
        try:
            safe_env = {
                "_getiter_": iter,
                "_getattr_": safer_getattr,
                "_getitem_": lambda obj, key: obj[key],
                "_write_": lambda obj: obj,
                "_inplacevar_": lambda op, x, y: op(x, y),
                "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
                "_print_": PrintCollector,
                "__builtins__": {
                    **limited_builtins,
                    **utility_builtins,
                    "len": len, "range": range, "str": str, "int": int,
                    "float": float, "list": list, "dict": dict,
                    "min": min, "max": max, "sorted": sorted, "enumerate": enumerate,
                },
                "context_list": context_list,
            }
            byte_code = compile_restricted_exec(edit_code)
            if byte_code.errors:
                return False, f"Compile errors: {byte_code.errors}"
            exec(byte_code.code, safe_env)
            return True, ""
        except Exception as e:
            return False, str(e)
    else:
        try:
            exec(edit_code, {"__builtins__": {}}, {"context_list": context_list})
            return True, ""
        except Exception as e:
            return False, str(e)


def _prefix_stability(before: list[str], after: list[str]) -> float:
    before_tokens = _tokenize(_serialize_context_list(before))
    after_tokens = _tokenize(_serialize_context_list(after))
    if not before_tokens:
        return 1.0
    common = 0
    for a, b in zip(before_tokens, after_tokens):
        if a == b:
            common += 1
        else:
            break
    return common / len(before_tokens)


def _parse_edit_operations(edit_code: str) -> list[dict]:
    ops = []
    if not edit_code:
        return ops
    for line in edit_code.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if re.match(r'context_list\.append\(', line):
            ops.append({"type": "append", "position": -1})
        elif m := re.match(r'context_list\.pop\((\d+)\)', line):
            ops.append({"type": "pop", "position": int(m.group(1))})
        elif re.match(r'context_list\.pop\(\s*\)', line):
            ops.append({"type": "pop", "position": -1})
        elif m := re.match(r'context_list\[(\d+)\]\s*=', line):
            ops.append({"type": "setitem", "position": int(m.group(1))})
        else:
            ops.append({"type": "other", "position": -1})
    return ops


# ============================================================
# LLM Judge (per-question-type, matching LongMemEval paper)
# ============================================================

def _get_judge_prompt(question_type: str, question: str, answer: str, response: str) -> str:
    """Build judge prompt matching LongMemEval's evaluate_qa.py."""
    is_abstention = question_type.endswith("_abs") or "abstention" in question_type.lower()

    if is_abstention:
        return (
            "I will give you an unanswerable question, an explanation, and a response from a model. "
            "Please answer yes if the model correctly identifies the question as unanswerable. "
            "The model could say that the information is incomplete, or some other information is given "
            "but the asked information is not.\n\n"
            f"Question: {question}\n\nExplanation: {answer}\n\nModel Response: {response}\n\n"
            "Does the model correctly identify the question as unanswerable? Answer yes or no only."
        )
    elif question_type == "temporal-reasoning":
        return (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response is equivalent to the correct answer or contains all the intermediate steps "
            "to get the correct answer, you should also answer yes. If the response only contains a subset "
            "of the information required by the answer, answer no. In addition, do not penalize off-by-one "
            "errors for the number of days. If the question asks for the number of days/weeks/months, etc., "
            "and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), "
            "the model's response is still correct.\n\n"
            f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    elif question_type == "knowledge-update":
        return (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response contains some previous information along with an updated answer, the response "
            "should be considered as correct as long as the updated answer is the required answer.\n\n"
            f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    elif question_type == "single-session-preference":
        return (
            "I will give you a question, a rubric for desired personalized response, and a response from a model. "
            "Please answer yes if the response satisfies the desired response. Otherwise, answer no. "
            "The model does not need to reflect all the points in the rubric. The response is correct as long "
            "as it recalls and utilizes the user's personal information correctly.\n\n"
            f"Question: {question}\n\nRubric: {answer}\n\nModel Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    else:
        # Default for single-session-user, single-session-assistant, multi-session
        return (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response is equivalent to the correct answer or contains all the intermediate steps "
            "to get the correct answer, you should also answer yes. If the response only contains a subset "
            "of the information required by the answer, answer no.\n\n"
            f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )


# ============================================================
# Context mode: multi-turn environment
# ============================================================

class LongMemEvalContextEnv(vf.MultiTurnEnv):
    """
    Multi-turn env for incremental context management on LongMemEval.

    Each rollout (no caching — every question has unique sessions):
      1. Process ~40 sessions one at a time with fresh prompts
      2. Answer the question from compressed context_list
      3. Score with LLM judge
    """

    def __init__(self, judge_model: str = "gpt-5.4", **kwargs):
        super().__init__(**kwargs)
        self.judge_model = judge_model

    async def setup_state(self, state: vf.State) -> vf.State:
        state = await super().setup_state(state)
        info = state.get("info", {})

        # Parse sessions from info
        sessions = json.loads(info["sessions_json"])
        dates = json.loads(info["dates_json"])

        state["_sessions"] = sessions
        state["_dates"] = dates
        state["_session_idx"] = 1  # session 0 is in initial prompt
        state["_context_list"] = []
        state["_question"] = info["question"]
        state["_question_date"] = info.get("question_date", "")
        state["_question_type"] = info.get("question_type", "")
        state["_question_sent"] = False
        state["_predicted_answer"] = ""
        state["_edit_log"] = []

        # Compute full history tokens for compression
        full_history = _format_all_sessions(sessions, dates)
        state["_full_history_tokens"] = _count_tokens(full_history)

        logger.info(
            f"[{info.get('question_id', '?')}] "
            f"Building context_list ({len(sessions)} sessions, "
            f"{state['_full_history_tokens']:,} full tokens)"
        )
        return state

    def _make_qa_prompt(self, state: vf.State) -> vf.Messages:
        question = state["_question"]
        question_date = state["_question_date"]
        return [
            SystemMessage(role="system", content=QA_SYSTEM_PROMPT),
            UserMessage(
                role="user",
                content=(
                    f"Facts:\n{_serialize_context_list(state['_context_list'])}\n\n"
                    f"Current Date: {question_date}\n"
                    f"Question: {question}\n\n"
                    f"Answer concisely using the provided facts. "
                    f"Put your answer in <answer> tags."
                ),
            ),
        ]

    async def get_prompt_messages(self, state: vf.State) -> vf.Messages:
        """Fresh prompts each turn (no message accumulation)."""
        if len(state["trajectory"]) == 0:
            return state["prompt"]

        prev_prompt = state["trajectory"][-1]["prompt"]
        prev_completion = state["trajectory"][-1]["completion"]
        messages = concat_messages([prev_prompt, prev_completion])
        env_resp = await self.env_response(messages, state)
        env_resp = maybe_normalize_messages(env_resp, field_name="env_response")

        if state.get("final_env_response") is not None:
            return concat_messages([messages, env_resp])

        if state["_question_sent"]:
            return self._make_qa_prompt(state)
        else:
            return [
                SystemMessage(role="system", content=CONTEXT_SYSTEM_PROMPT),
                *env_resp,
            ]

    async def env_response(
        self, messages: vf.Messages, state: vf.State, **kwargs
    ) -> vf.Messages:
        last_msg = messages[-1]
        content = last_msg.content if hasattr(last_msg, "content") else ""

        if state["_question_sent"]:
            predicted = _extract_answer_tag(content)
            state["_predicted_answer"] = predicted
            # Aggregate metrics for answer step
            ctx_tokens = _count_tokens(_serialize_context_list(state["_context_list"]))
            full_tokens = state.get("_full_history_tokens", 0)
            compression = (ctx_tokens / full_tokens * 100) if full_tokens > 0 else 0
            ps_scores = [e.get("prefix_stability") for e in state["_edit_log"] if "prefix_stability" in e]
            avg_ps = sum(ps_scores) / len(ps_scores) if ps_scores else None
            all_ops = []
            for e in state["_edit_log"]:
                all_ops.extend(e.get("edit_ops", []))
            op_counts = {}
            for op in all_ops:
                op_counts[op["type"]] = op_counts.get(op["type"], 0) + 1
            max_list_len = max((e.get("context_list_len", 0) for e in state["_edit_log"]), default=0)
            append_ratio = (op_counts.get("append", 0) / len(all_ops)) if all_ops else 0.0

            state["_edit_log"].append({
                "step": "answer",
                "predicted_answer": predicted,
                "final_context_list_len": len(state["_context_list"]),
                "context_tokens": ctx_tokens,
                "full_history_tokens": full_tokens,
                "compression_pct": round(compression, 1),
                "avg_prefix_stability": avg_ps,
                "edit_op_distribution": op_counts,
                "total_edits": len(all_ops),
                "max_list_length": max_list_len,
                "append_ratio": append_ratio,
            })
            state["_context_tokens"] = ctx_tokens
            state["_full_history_tokens"] = full_tokens

            logger.info(
                f"Answer: {predicted[:80]} | ctx_len={len(state['_context_list'])} | "
                f"ctx_tok={ctx_tokens} | compression={compression:.1f}%"
            )
            final = [UserMessage(role="user", content="[Evaluation complete]")]
            state["final_env_response"] = final
            return final

        # Snapshot + process edit
        before = list(state["_context_list"])
        edit_code = _extract_edit_code(content)
        edit_success = True
        edit_error = ""
        if edit_code:
            edit_success, edit_error = _execute_context_edit(edit_code, state["_context_list"])
            if not edit_success:
                state["_context_list"] = list(before)

        session_idx = state["_session_idx"]
        ctx_tokens = _count_tokens(_serialize_context_list(state["_context_list"]))
        ps = _prefix_stability(before, state["_context_list"])
        edit_ops = _parse_edit_operations(edit_code) if edit_success else []

        state["_edit_log"].append({
            "step": session_idx - 1,
            "edit_code": edit_code,
            "edit_success": edit_success,
            "edit_error": edit_error if not edit_success else "",
            "edit_ops": edit_ops,
            "context_list_len": len(state["_context_list"]),
            "context_tokens": ctx_tokens,
            "prefix_stability": ps,
        })
        logger.info(
            f"Session {session_idx - 1}/{len(state['_sessions'])} | "
            f"edit_ok={edit_success} | ctx_len={len(state['_context_list'])} | "
            f"ctx_tok={ctx_tokens} | prefix={ps:.2f} | ops={len(edit_ops)}"
        )

        sessions = state["_sessions"]
        dates = state["_dates"]

        if session_idx < len(sessions):
            dialogue = _format_session(sessions[session_idx], dates[session_idx] if session_idx < len(dates) else "")
            state["_session_idx"] = session_idx + 1
            return [
                UserMessage(
                    role="user",
                    content=(
                        f"Current context list: {_serialize_context_list(state['_context_list'])}\n\n"
                        f"New dialogue:\n{dialogue}"
                    ),
                )
            ]
        else:
            state["_question_sent"] = True
            logger.info(f"All sessions done, asking: {state['_question'][:60]}...")
            return [
                UserMessage(role="user", content="[Sessions complete]")
            ]

    async def add_trajectory_step(self, state: vf.State, trajectory_step):
        """Only add edit turns to trajectory — QA turn generates reward but no gradients."""
        if state.get("_question_sent"):
            state["_qa_completion"] = trajectory_step["completion"]
            return
        state["trajectory"].append(trajectory_step)

    async def render_completion(self, state: vf.State):
        """Save full trajectory + ensure state columns are in the dict for export."""
        completion = []
        for step in state["trajectory"]:
            completion.extend(step.get("completion", []))
        if state.get("_qa_completion"):
            completion.extend(state["_qa_completion"])
        state["completion"] = completion
        # Force state columns into the dict so --state-columns can find them
        dict.__setitem__(state, "_context_list", state.get("_context_list", []))
        dict.__setitem__(state, "_edit_log", state.get("_edit_log", []))
        dict.__setitem__(state, "_predicted_answer", state.get("_predicted_answer", ""))


# ============================================================
# Compaction mode: one-shot compression then QA (2 turns)
# ============================================================

class LongMemEvalCompactionEnv(vf.MultiTurnEnv):
    """
    One-shot compaction: full history → model compresses into context_list → QA.
    No caching — each question has unique sessions.
    """

    def __init__(self, judge_model: str = "gpt-5.4", **kwargs):
        super().__init__(**kwargs)
        self.judge_model = judge_model

    async def setup_state(self, state: vf.State) -> vf.State:
        state = await super().setup_state(state)
        info = state.get("info", {})

        state["_question"] = info["question"]
        state["_question_date"] = info.get("question_date", "")
        state["_question_type"] = info.get("question_type", "")
        state["_context_list"] = []
        state["_edit_log"] = []
        state["_predicted_answer"] = ""
        state["_question_sent"] = False

        # Compute full history tokens
        sessions = json.loads(info["sessions_json"])
        dates = json.loads(info["dates_json"])
        full_history = _format_all_sessions(sessions, dates)
        state["_full_history_tokens"] = _count_tokens(full_history)

        return state

    def _make_qa_prompt(self, state: vf.State) -> vf.Messages:
        question = state["_question"]
        question_date = state["_question_date"]
        return [
            SystemMessage(role="system", content=QA_SYSTEM_PROMPT),
            UserMessage(
                role="user",
                content=(
                    f"Facts:\n{_serialize_context_list(state['_context_list'])}\n\n"
                    f"Current Date: {question_date}\n"
                    f"Question: {question}\n\n"
                    f"Answer concisely using the provided facts. "
                    f"Put your answer in <answer> tags."
                ),
            ),
        ]

    async def get_prompt_messages(self, state: vf.State) -> vf.Messages:
        if len(state["trajectory"]) == 0:
            return state["prompt"]

        prev_prompt = state["trajectory"][-1]["prompt"]
        prev_completion = state["trajectory"][-1]["completion"]
        messages = concat_messages([prev_prompt, prev_completion])
        env_resp = await self.env_response(messages, state)
        env_resp = maybe_normalize_messages(env_resp, field_name="env_response")

        if state.get("final_env_response") is not None:
            return concat_messages([messages, env_resp])

        if state["_question_sent"]:
            return self._make_qa_prompt(state)
        else:
            return concat_messages([messages, env_resp])

    async def env_response(self, messages: vf.Messages, state: vf.State, **kwargs) -> vf.Messages:
        last_msg = messages[-1]
        content = last_msg.content if hasattr(last_msg, "content") else ""

        if state["_question_sent"]:
            predicted = _extract_answer_tag(content)
            state["_predicted_answer"] = predicted
            ctx_tokens = _count_tokens(_serialize_context_list(state["_context_list"]))
            full_tokens = state.get("_full_history_tokens", 0)
            compression = (ctx_tokens / full_tokens * 100) if full_tokens > 0 else 0
            state["_edit_log"].append({
                "step": "answer",
                "predicted_answer": predicted,
                "final_context_list_len": len(state["_context_list"]),
                "context_tokens": ctx_tokens,
                "full_history_tokens": full_tokens,
                "compression_pct": round(compression, 1),
            })
            state["_context_tokens"] = ctx_tokens
            state["_full_history_tokens"] = full_tokens
            logger.info(f"Answer: {predicted[:80]} | compression={compression:.1f}%")
            final = [UserMessage(role="user", content="[Evaluation complete]")]
            state["final_env_response"] = final
            return final

        # Process compaction response
        edit_code = _extract_edit_code(content)
        if edit_code:
            _execute_context_edit(edit_code, state["_context_list"])

        ctx_tokens = _count_tokens(_serialize_context_list(state["_context_list"]))
        state["_edit_log"].append({
            "step": "compaction",
            "edit_code": edit_code,
            "context_list_len": len(state["_context_list"]),
            "context_tokens": ctx_tokens,
        })
        logger.info(f"Compacted to {len(state['_context_list'])} items ({ctx_tokens} tokens)")

        state["_question_sent"] = True
        return [UserMessage(role="user", content="[Compaction complete]")]

    async def render_completion(self, state: vf.State):
        completion = []
        for step in state["trajectory"]:
            completion.extend(step.get("completion", []))
        state["completion"] = completion
        dict.__setitem__(state, "_context_list", state.get("_context_list", []))
        dict.__setitem__(state, "_edit_log", state.get("_edit_log", []))
        dict.__setitem__(state, "_predicted_answer", state.get("_predicted_answer", ""))


# ============================================================
# Reward function (LLM judge)
# ============================================================

def _judge_reward_sync(question: str, answer: str, response: str,
                       question_type: str, judge_model: str) -> float:
    """Call judge model to score response. Returns 1.0 (correct) or 0.0 (incorrect)."""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    prompt = _get_judge_prompt(question_type, question, answer, response)
    try:
        result = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=20,
            temperature=0.0,
        )
        judge_response = result.choices[0].message.content.strip().lower()
        return 1.0 if "yes" in judge_response else 0.0
    except Exception as e:
        logger.error(f"Judge call failed: {e}")
        return 0.0


class LongMemEvalRubric(vf.Rubric):
    def __init__(self, judge_model: str = "gpt-5.4"):
        super().__init__()
        self.judge_model = judge_model
        self.add_reward_func(self.judge_score, weight=1.0)

    async def judge_score(self, state: vf.State) -> float:
        info = state.get("info", {})
        answer = info.get("answer", "")
        question = info.get("question", "")
        question_type = info.get("question_type", "")

        # Extract predicted from completion
        completion = state.get("completion", [])
        if completion:
            last = completion[-1]
            raw = last.content if hasattr(last, "content") else last.get("content", "")
            predicted = _extract_answer_tag(raw.strip())
        else:
            predicted = ""

        if not predicted:
            return 0.0

        # Run judge in thread (sync OpenAI call)
        import asyncio
        loop = asyncio.get_running_loop()
        score = await loop.run_in_executor(
            None,
            _judge_reward_sync,
            question, str(answer), predicted, question_type, self.judge_model,
        )
        return score


# ============================================================
# Context metrics (same pattern as locomo)
# ============================================================

def prefix_stability_metric(state: vf.State) -> float:
    edit_log = state.get("_edit_log", [])
    scores = [e.get("prefix_stability") for e in edit_log if "prefix_stability" in e]
    return sum(scores) / len(scores) if scores else 0.0


def compression_metric(state: vf.State) -> float:
    ctx_tok = state.get("_context_tokens", 0)
    full_tok = state.get("_full_history_tokens", 0)
    if full_tok == 0 or ctx_tok == 0:
        return 0.0
    return ctx_tok / full_tok


def context_tokens_metric(state: vf.State) -> float:
    return float(state.get("_context_tokens", 0))


def append_ratio_metric(state: vf.State) -> float:
    edit_log = state.get("_edit_log", [])
    all_ops = []
    for e in edit_log:
        all_ops.extend(e.get("edit_ops", []))
    if not all_ops:
        return 0.0
    return sum(1 for op in all_ops if op["type"] == "append") / len(all_ops)


def total_edits_metric(state: vf.State) -> float:
    edit_log = state.get("_edit_log", [])
    return float(sum(len(e.get("edit_ops", [])) for e in edit_log))


def max_list_length_metric(state: vf.State) -> float:
    edit_log = state.get("_edit_log", [])
    return float(max((e.get("context_list_len", 0) for e in edit_log), default=0))


# ============================================================
# Data download + dataset builder
# ============================================================

DATA_URL = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"
DATA_DIR = Path(os.environ.get("LONGMEMEVAL_DATA_DIR", Path.home() / ".cache" / "longmemeval"))


def _download_data(variant: str = "longmemeval_s_cleaned") -> Path:
    """Download LongMemEval data if not cached."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{variant}.json"
    filepath = DATA_DIR / filename
    if not filepath.exists():
        url = f"{DATA_URL}/{filename}"
        logger.info(f"Downloading {url} to {filepath}...")
        urllib.request.urlretrieve(url, filepath)
        logger.info(f"Downloaded {filepath}")
    return filepath


def _build_dataset(mode: str, variant: str, n: int, question_ids: list[str] | None = None):
    def builder():
        filepath = _download_data(variant)
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        if question_ids:
            qid_set = set(question_ids)
            data = [e for e in data if e["question_id"] in qid_set]
        if n > 0:
            data = data[:n]

        if mode == "baseline":
            rows = []
            for entry in data:
                sessions = entry["haystack_sessions"]
                dates = entry.get("haystack_dates", [""] * len(sessions))
                history = _format_all_sessions(sessions, dates)
                question_date = entry.get("question_date", "")
                rows.append({
                    "prompt": [
                        {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                f"I will give you several history chats between you and a user. "
                                f"Please answer the question based on the relevant chat history.\n\n"
                                f"History Chats:\n\n{history}\n\n"
                                f"Current Date: {question_date}\n"
                                f"Question: {entry['question']}\n"
                                f"Answer:"
                            ),
                        },
                    ],
                    "answer": str(entry.get("answer", "")),
                    "info": json.dumps({
                        "question_id": entry["question_id"],
                        "question": entry["question"],
                        "answer": str(entry.get("answer", "")),
                        "question_type": entry.get("question_type", ""),
                        "question_date": entry.get("question_date", ""),
                    }),
                })
            return Dataset.from_list(rows)

        elif mode == "compaction":
            rows = []
            for entry in data:
                sessions = entry["haystack_sessions"]
                dates = entry.get("haystack_dates", [""] * len(sessions))
                history = _format_all_sessions(sessions, dates)
                rows.append({
                    "prompt": [
                        {"role": "system", "content": COMPACTION_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                f"Below is a conversation history between a user and an assistant.\n\n"
                                f"{history}\n\n"
                                f"Extract all important facts into context_list using <edit> tags."
                            ),
                        },
                    ],
                    "answer": str(entry.get("answer", "")),
                    "info": json.dumps({
                        "question_id": entry["question_id"],
                        "question": entry["question"],
                        "answer": str(entry.get("answer", "")),
                        "question_type": entry.get("question_type", ""),
                        "question_date": entry.get("question_date", ""),
                        "sessions_json": json.dumps(sessions),
                        "dates_json": json.dumps(dates),
                    }),
                })
            return Dataset.from_list(rows)

        else:  # context mode
            rows = []
            for entry in data:
                sessions = entry["haystack_sessions"]
                dates = entry.get("haystack_dates", [""] * len(sessions))
                first_session = _format_session(sessions[0], dates[0] if dates else "")
                rows.append({
                    "prompt": [
                        {"role": "system", "content": CONTEXT_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                f"Current context list: []\n\n"
                                f"New dialogue:\n{first_session}"
                            ),
                        },
                    ],
                    "answer": str(entry.get("answer", "")),
                    "info": json.dumps({
                        "question_id": entry["question_id"],
                        "question": entry["question"],
                        "answer": str(entry.get("answer", "")),
                        "question_type": entry.get("question_type", ""),
                        "question_date": entry.get("question_date", ""),
                        "sessions_json": json.dumps(sessions),
                        "dates_json": json.dumps(dates),
                    }),
                })
            return Dataset.from_list(rows)

    return builder


# ============================================================
# Entry point
# ============================================================

def load_environment(
    mode: str = "context",
    variant: str = "longmemeval_s_cleaned",
    num_examples: int = -1,
    question_ids: str = "",
    judge_model: str = "gpt-5.4",
) -> vf.Environment:
    """
    Load the LongMemEval context management environment.

    Args:
        mode: "baseline" (full history per question) or "context" (incremental fact management)
        variant: "longmemeval_s_cleaned" (~40 sessions), "longmemeval_m_cleaned" (~500 sessions),
                 or "longmemeval_oracle" (oracle retrieval)
        num_examples: Limit number of examples (-1 for all)
        question_ids: Comma-separated question IDs to filter (empty for all)
        judge_model: Model for LLM judge scoring
    """
    qids = [q.strip() for q in question_ids.split(",") if q.strip()] if question_ids else None
    dataset_builder = _build_dataset(mode, variant, num_examples, qids)
    rubric = LongMemEvalRubric(judge_model=judge_model)

    if mode == "baseline":
        return vf.SingleTurnEnv(
            dataset=dataset_builder,
            rubric=rubric,
        )
    elif mode == "context":
        env = LongMemEvalContextEnv(
            dataset=dataset_builder,
            rubric=rubric,
            judge_model=judge_model,
            max_turns=600,  # up to 500 sessions + QA + buffer
        )
        env.rubric.add_metric(prefix_stability_metric, weight=0.0)
        env.rubric.add_metric(compression_metric, weight=0.0)
        env.rubric.add_metric(context_tokens_metric, weight=0.0)
        env.rubric.add_metric(append_ratio_metric, weight=0.0)
        env.rubric.add_metric(total_edits_metric, weight=0.0)
        env.rubric.add_metric(max_list_length_metric, weight=0.0)
        return env
    elif mode == "compaction":
        env = LongMemEvalCompactionEnv(
            dataset=dataset_builder,
            rubric=rubric,
            judge_model=judge_model,
            max_turns=5,  # compaction turn + QA turn + buffer
        )
        env.rubric.add_metric(compression_metric, weight=0.0)
        env.rubric.add_metric(context_tokens_metric, weight=0.0)
        return env
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'baseline', 'context', or 'compaction'.")
