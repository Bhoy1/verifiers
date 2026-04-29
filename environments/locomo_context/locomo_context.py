"""
LoCoMo Context Management Environment

Evaluates LLM long-conversation memory via the LoCoMo benchmark.
Two modes:
  - baseline: Full conversation history dumped per question (SingleTurnEnv)
  - context:  Incremental session-by-session fact extraction into a context list,
              then answer from compressed facts (MultiTurnEnv)

Context mode matches the original eval.py flow:
  1. Process all sessions ONCE per conversation (fresh prompt each turn, no history accumulation)
  2. Cache the context_list
  3. Answer all questions as single-turn QA from the cached context_list

Scoring: token-level F1 (same as original LoCoMo paper).
"""

import asyncio
import json
import logging
import re
from collections import Counter

from datasets import Dataset, load_dataset

_enc = None
try:
    import tiktoken
    # Try newest encodings first (gpt-5.x, gpt-4o use o200k_base; gpt-4 uses cl100k_base)
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

import verifiers as vf
from verifiers.types import SystemMessage, UserMessage
from verifiers.utils.message_utils import concat_messages, maybe_normalize_messages
from verifiers.utils.usage_utils import extract_usage_tokens

logger = logging.getLogger(__name__)


# ============================================================
# System prompts
# ============================================================

CONTEXT_SYSTEM_PROMPT = """You manage a persistent context list that stores key facts from an ongoing conversation between two people.

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

Example:
  Current context list: ["Caroline attends LGBTQ support group", "Melanie has 2 kids"]
  New dialogue:
  Caroline: I just got accepted to the nursing program at UCLA!

  <edit>context_list.append("Caroline accepted to nursing program at UCLA")</edit>"""

BASELINE_SYSTEM_PROMPT = """You are a helpful assistant observing a multi-session conversation between two people. The conversation takes place over multiple days and the date of each conversation is written at the beginning of the conversation. Pay close attention to all details, facts, dates, and events mentioned."""

QA_SYSTEM_PROMPT = "Answer the question using the provided facts. Write an answer in the form of a short phrase. Answer with exact words from the facts whenever possible."

QA_PROMPT = """Based on the above context, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {} Short answer:"""

QA_PROMPT_CAT_5 = """Based on the above context, answer the following question. If the information is not available in the context, say 'No information available'.

Question: {} Short answer:"""

COMPACTION_SYSTEM_PROMPT = """You are given a conversation history. Extract all important facts into a Python list called context_list. Save key details: names, dates, locations, preferences, decisions, specific values, sentiments, and attitudes.

Output your context_list inside <edit> tags:
<edit>
context_list.append("fact 1")
context_list.append("fact 2")
...
</edit>"""

# v3 prompt — softer, priority-based, encourages pop/update
CONTEXT_SYSTEM_PROMPT_V3 = """You manage a persistent context list that stores key facts from an ongoing conversation between two people.

Every turn, output an <edit> block to update your context list.

Format:
<edit>
context_list.append("new fact")
context_list[i] = "updated fact"
context_list.pop(i)
</edit>

What to save (in priority order):
1. Specific details — names, dates, times, locations, exact numbers
2. Preferences, decisions, attitudes, and sentiments
3. Key events and changes in circumstances

Context hygiene:
- When information changes, UPDATE the old entry — don't leave contradictory items
- When the list gets long, compress related entries or pop the least important ones"""

# Minimal prompt — bare bones, let RL discover the strategy
CONTEXT_SYSTEM_PROMPT_MINIMAL = """You have limited memory. Use <edit> blocks to manage your context list.

<edit>
context_list.append("new fact")
context_list[i] = "updated fact"
context_list.pop(i)
</edit>"""

# Map prompt versions to prompts
PROMPT_VERSIONS = {
    "v1": CONTEXT_SYSTEM_PROMPT,
    "v3": CONTEXT_SYSTEM_PROMPT_V3,
    "minimal": CONTEXT_SYSTEM_PROMPT_MINIMAL,
}


# ============================================================
# Conversation parsing helpers
# ============================================================

def _get_sessions(conversation: dict) -> list[tuple[str, str, list[dict]]]:
    sessions = []
    i = 1
    while f"session_{i}" in conversation:
        key = f"session_{i}"
        date = conversation.get(f"session_{i}_date_time", "")
        sessions.append((key, date, conversation[key]))
        i += 1
    return sessions


def _format_session(turns: list[dict], date: str = "") -> str:
    lines = []
    if date:
        lines.append(f"DATE: {date}")
        lines.append("CONVERSATION:")
    for turn in turns:
        speaker = turn.get("speaker", "Unknown")
        text = turn.get("text", "").strip()
        lines.append(f'{speaker} said, "{text}"')
    return "\n".join(lines)


def _format_all_sessions(sessions: list[tuple]) -> str:
    parts = []
    for i, (key, date, turns) in enumerate(sessions):
        parts.append(f"### Session {i + 1}:\n{_format_session(turns, date)}")
    return "\n\n".join(parts)


# ============================================================
# Context list helpers
# ============================================================

def _serialize_context_list(context_list: list[str]) -> str:
    if not context_list:
        return "[]"
    items = ", ".join(f'"{item}"' for item in context_list)
    return f"[{items}]"


def _parse_edit_operations(edit_code: str) -> list[dict]:
    """Parse edit_code into {type, position} ops. position=-1 for appends."""
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


def _prefix_stability(before: list[str], after: list[str]) -> float:
    """
    Fraction of 'before' tokens preserved at start of 'after'.
    1.0 = perfect prefix reuse (just appending), 0.0 = totally rewritten.
    """
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


def _extract_edit_code(response: str) -> str | None:
    match = re.search(r"<edit>(.*?)</edit>", response, re.DOTALL)
    if not match:
        return None
    code = match.group(1).strip()
    # Strip "context_list = []" reassignment — model sometimes resets the list
    # which creates a new local variable instead of modifying the passed-in list
    code = re.sub(r'^context_list\s*=\s*\[\s*\]\s*\n?', '', code, flags=re.MULTILINE)
    return code.strip() if code.strip() else None


def _extract_answer_tag(response: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: strip any <edit> blocks and return remaining text
    text = re.sub(r"<edit>.*?</edit>", "", response, flags=re.DOTALL).strip()
    return text


try:
    from RestrictedPython import compile_restricted_exec, limited_builtins, utility_builtins
    from RestrictedPython.Guards import guarded_iter_unpack_sequence, safer_getattr
    from RestrictedPython.PrintCollector import PrintCollector
    _HAS_RESTRICTED_PYTHON = True
except ImportError:
    _HAS_RESTRICTED_PYTHON = False


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
                    "len": len,
                    "range": range,
                    "str": str,
                    "int": int,
                    "float": float,
                    "list": list,
                    "dict": dict,
                    "min": min,
                    "max": max,
                    "sorted": sorted,
                    "enumerate": enumerate,
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


# ============================================================
# F1 scoring (matches original LoCoMo evaluation)
# ============================================================

def _f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = prediction.lower().split()
    gold_tokens = ground_truth.lower().split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ============================================================
# Context list cache (shared across rollouts)
# ============================================================

_context_cache: dict[str, list[str]] = {}
_cache_edit_logs: dict[str, list[dict]] = {}
_cache_events: dict[str, asyncio.Event] = {}
_cache_full_history_tokens: dict[str, int] = {}


# ============================================================
# Context mode: multi-turn environment
# ============================================================

class LoCoMoContextEnv(vf.MultiTurnEnv):
    """
    Multi-turn env for incremental context management.

    Matches the original eval.py flow:
      1. First rollout per conversation processes all sessions → builds context_list
      2. Context_list is cached; subsequent rollouts skip straight to QA
      3. Each session turn uses a FRESH prompt (no message accumulation)
      4. QA is single-turn from the cached/built context_list
    """

    async def setup_state(self, state: vf.State) -> vf.State:
        state = await super().setup_state(state)
        info = state.get("info", {})
        conv_id = info["conv_id"]

        state["_question"] = info["question"]
        state["_category"] = info["category"]
        state["_conv_id"] = conv_id
        state["_predicted_answer"] = ""
        state["_edit_log"] = []

        if conv_id in _context_cache:
            # Already built — skip straight to QA
            state["_context_list"] = list(_context_cache[conv_id])
            state["_edit_log"] = list(_cache_edit_logs.get(conv_id, []))
            state["_full_history_tokens"] = _cache_full_history_tokens.get(conv_id, 0)
            state["_skip_sessions"] = True
            state["_question_sent"] = True  # first turn will be QA
            logger.info(
                f"[{conv_id}] Using cached context_list "
                f"({len(state['_context_list'])} items) for: {info['question'][:50]}..."
            )
        elif conv_id in _cache_events:
            # Another rollout is building — wait for it
            logger.info(f"[{conv_id}] Waiting for context_list to be built...")
            await _cache_events[conv_id].wait()
            state["_context_list"] = list(_context_cache[conv_id])
            state["_edit_log"] = list(_cache_edit_logs.get(conv_id, []))
            state["_full_history_tokens"] = _cache_full_history_tokens.get(conv_id, 0)
            state["_skip_sessions"] = True
            state["_question_sent"] = True
            logger.info(
                f"[{conv_id}] Got cached context_list "
                f"({len(state['_context_list'])} items) for: {info['question'][:50]}..."
            )
        else:
            # First rollout for this conversation — process sessions
            _cache_events[conv_id] = asyncio.Event()
            conversation = json.loads(info["conversation"])
            sessions = _get_sessions(conversation)
            # Compute full history tokens for compression ratio
            full_history = _format_all_sessions(sessions)
            state["_full_history_tokens"] = _count_tokens(full_history)
            state["_sessions"] = sessions
            state["_session_idx"] = 1  # session 0 is in the initial prompt
            state["_context_list"] = []
            state["_skip_sessions"] = False
            state["_question_sent"] = False
            logger.info(
                f"[{conv_id}] Building context_list "
                f"({len(sessions)} sessions)"
            )

        return state

    def _make_qa_prompt(self, state: vf.State) -> vf.Messages:
        question = state["_question"]
        category = state["_category"]

        # Per-category question formatting (matches original LoCoMo)
        if category == 2:
            question = question + " Use DATE of CONVERSATION to answer with an approximate date."

        if category == 5:
            qa_prompt = QA_PROMPT_CAT_5.format(question)
        else:
            qa_prompt = QA_PROMPT.format(question)

        return [
            SystemMessage(role="system", content=QA_SYSTEM_PROMPT),
            UserMessage(
                role="user",
                content=(
                    f"Facts:\n{_serialize_context_list(state['_context_list'])}\n\n"
                    f"{qa_prompt}\n\n"
                    f"Put your answer in <answer> tags."
                ),
            ),
        ]

    async def get_prompt_messages(self, state: vf.State) -> vf.Messages:
        """Override to send FRESH prompts each turn (no message accumulation)."""
        if len(state["trajectory"]) == 0:
            if state.get("_skip_sessions"):
                # Cached context — go straight to QA
                return self._make_qa_prompt(state)
            # First session is in the initial prompt
            return state["prompt"]

        # Process previous model response via env_response
        prev_prompt = state["trajectory"][-1]["prompt"]
        prev_completion = state["trajectory"][-1]["completion"]
        messages = concat_messages([prev_prompt, prev_completion])
        env_resp = await self.env_response(messages, state)
        env_resp = maybe_normalize_messages(env_resp, field_name="env_response")

        if state.get("final_env_response") is not None:
            # Done — return accumulated for render_completion
            return concat_messages([messages, env_resp])

        if state["_question_sent"]:
            # QA turn — fresh prompt with context_list + question
            return self._make_qa_prompt(state)
        else:
            # Next session — fresh prompt (system + session user message)
            return [
                SystemMessage(role="system", content=CONTEXT_SYSTEM_PROMPT),
                *env_resp,
            ]

    async def env_response(
        self, messages: vf.Messages, state: vf.State, **kwargs
    ) -> vf.Messages:
        last_msg = messages[-1]
        content = (last_msg.content if hasattr(last_msg, "content") else "") or ""

        if state["_question_sent"]:
            # Model just answered the question — extract answer and finish
            predicted = _extract_answer_tag(content)
            state["_predicted_answer"] = predicted
            ctx_tokens = _count_tokens(_serialize_context_list(state["_context_list"]))
            full_tokens = state.get("_full_history_tokens", 0)
            compression = (ctx_tokens / full_tokens * 100) if full_tokens > 0 else 0
            # Get tokens from last API call
            in_tok, out_tok = 0, 0
            if state["trajectory"]:
                resp = state["trajectory"][-1].get("response")
                if resp is not None:
                    in_tok, out_tok = extract_usage_tokens(resp)
            # Aggregate across session steps
            ps_scores = [
                e.get("prefix_stability") for e in state["_edit_log"]
                if "prefix_stability" in e
            ]
            avg_ps = sum(ps_scores) / len(ps_scores) if ps_scores else None
            # Edit operation aggregates
            all_ops = []
            for e in state["_edit_log"]:
                all_ops.extend(e.get("edit_ops", []))
            op_counts = {}
            for op in all_ops:
                op_counts[op["type"]] = op_counts.get(op["type"], 0) + 1
            max_list_len = max(
                (e.get("context_list_len", 0) for e in state["_edit_log"]),
                default=0,
            )
            append_ratio = (op_counts.get("append", 0) / len(all_ops)) if all_ops else 0.0
            state["_edit_log"].append({
                "step": "answer",
                "predicted_answer": predicted,
                "final_context_list": list(state["_context_list"]),
                "final_context_list_len": len(state["_context_list"]),
                "context_tokens": ctx_tokens,
                "full_history_tokens": full_tokens,
                "compression_pct": round(compression, 1),
                "avg_prefix_stability": avg_ps,
                "edit_op_distribution": op_counts,
                "total_edits": len(all_ops),
                "edit_positions": [op["position"] for op in all_ops if op["position"] >= 0],
                "max_list_length": max_list_len,
                "append_ratio": append_ratio,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
            })
            # Store for metrics
            state["_append_ratio"] = append_ratio
            state["_total_edits"] = len(all_ops)
            state["_max_list_length"] = max_list_len
            state["_context_tokens"] = ctx_tokens
            logger.info(
                f"[{state['_conv_id']}] "
                f"Answer: {predicted[:80]} | ctx_len={len(state['_context_list'])} | ctx_tok={ctx_tokens}"
            )
            final = [UserMessage(role="user", content="[Evaluation complete]")]
            state["final_env_response"] = final
            return final

        # Snapshot before edit (for prefix stability)
        before = list(state["_context_list"])

        # Process <edit> from model's response
        edit_code = _extract_edit_code(content)
        edit_success = True
        edit_error = ""
        if edit_code:
            ctx = state["_context_list"]
            edit_success, edit_error = _execute_context_edit(edit_code, ctx)
            if not edit_success:
                state["_context_list"] = list(before)

        # Log this edit step
        session_idx = state["_session_idx"]
        ctx_tokens = _count_tokens(_serialize_context_list(state["_context_list"]))
        ps = _prefix_stability(before, state["_context_list"])
        edit_ops = _parse_edit_operations(edit_code) if edit_success else []
        # Get tokens from last API call
        in_tok, out_tok = 0, 0
        if state["trajectory"]:
            resp = state["trajectory"][-1].get("response")
            if resp is not None:
                in_tok, out_tok = extract_usage_tokens(resp)
        state["_edit_log"].append({
            "step": session_idx - 1,
            "edit_code": edit_code,
            "edit_success": edit_success,
            "edit_error": edit_error if not edit_success else "",
            "edit_ops": edit_ops,
            "context_list": list(state["_context_list"]),
            "context_list_len": len(state["_context_list"]),
            "context_tokens": ctx_tokens,
            "prefix_stability": ps,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
        })
        logger.info(
            f"[{state['_conv_id']}] "
            f"Session {session_idx - 1}/{len(state['_sessions'])} | "
            f"edit_ok={edit_success} | ctx_len={len(state['_context_list'])} | "
            f"ctx_tok={ctx_tokens} | prefix={ps:.2f} | in={in_tok} out={out_tok}"
        )

        sessions = state["_sessions"]

        if session_idx < len(sessions):
            # Send next session
            _key, date, turns = sessions[session_idx]
            dialogue = _format_session(turns, date)
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
            # All sessions done — cache context_list and proceed to QA
            conv_id = state["_conv_id"]
            _context_cache[conv_id] = list(state["_context_list"])
            _cache_edit_logs[conv_id] = list(state["_edit_log"])
            _cache_full_history_tokens[conv_id] = state.get("_full_history_tokens", 0)
            if conv_id in _cache_events:
                _cache_events[conv_id].set()  # wake up waiting rollouts
            ctx_tokens = _count_tokens(_serialize_context_list(state["_context_list"]))
            full_tokens = state.get("_full_history_tokens", 0)
            compression = (ctx_tokens / full_tokens * 100) if full_tokens > 0 else 0
            logger.info(
                f"[{conv_id}] Context list cached "
                f"({len(state['_context_list'])} items) | "
                f"ctx_tok={ctx_tokens} / full_tok={full_tokens} → {compression:.1f}% compression"
            )

            state["_question_sent"] = True
            # Return placeholder — get_prompt_messages will use _make_qa_prompt instead
            return [
                UserMessage(
                    role="user",
                    content="[Sessions complete]",
                )
            ]

    async def add_trajectory_step(self, state: vf.State, trajectory_step):
        """Only add edit turns to trajectory — QA turn generates reward but no gradients."""
        if state.get("_question_sent"):
            state["_qa_completion"] = trajectory_step["completion"]
            return
        state["trajectory"].append(trajectory_step)

    async def render_completion(self, state: vf.State):
        """Save full trajectory: all assistant responses across turns (session edits + QA answer)."""
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

_compaction_cache: dict[str, list[str]] = {}
_compaction_events: dict[str, asyncio.Event] = {}
_compaction_full_tokens: dict[str, int] = {}


class LoCoMoCompactionEnv(vf.MultiTurnEnv):
    """
    One-shot compaction: dump full history → model compresses into context_list → QA.

    Turn 1: Full conversation history → model outputs <edit> block with all facts
    Turn 2: QA from the compacted context_list

    Caches compaction per conversation (like context mode).
    """

    async def setup_state(self, state: vf.State) -> vf.State:
        state = await super().setup_state(state)
        info = state.get("info", {})
        conv_id = info["conv_id"]

        state["_question"] = info["question"]
        state["_category"] = info["category"]
        state["_conv_id"] = conv_id
        state["_predicted_answer"] = ""
        state["_context_list"] = []
        state["_edit_log"] = []
        state["_question_sent"] = False

        if conv_id in _compaction_cache:
            state["_context_list"] = list(_compaction_cache[conv_id])
            state["_skip_compaction"] = True
            state["_question_sent"] = True
            state["_full_history_tokens"] = _compaction_full_tokens.get(conv_id, 0)
            logger.info(f"[{conv_id}] Using cached compaction ({len(state['_context_list'])} items)")
        elif conv_id in _compaction_events:
            logger.info(f"[{conv_id}] Waiting for compaction...")
            await _compaction_events[conv_id].wait()
            state["_context_list"] = list(_compaction_cache[conv_id])
            state["_skip_compaction"] = True
            state["_question_sent"] = True
            state["_full_history_tokens"] = _compaction_full_tokens.get(conv_id, 0)
        else:
            _compaction_events[conv_id] = asyncio.Event()
            conversation = json.loads(info["conversation"])
            sessions = _get_sessions(conversation)
            full_history = _format_all_sessions(sessions)
            state["_full_history"] = full_history
            state["_full_history_tokens"] = _count_tokens(full_history)
            state["_skip_compaction"] = False

        return state

    def _make_qa_prompt(self, state: vf.State) -> vf.Messages:
        question = state["_question"]
        category = state["_category"]
        if category == 2:
            question = question + " Use DATE of CONVERSATION to answer with an approximate date."
        if category == 5:
            qa_prompt = QA_PROMPT_CAT_5.format(question)
        else:
            qa_prompt = QA_PROMPT.format(question)
        return [
            SystemMessage(role="system", content=QA_SYSTEM_PROMPT),
            UserMessage(
                role="user",
                content=(
                    f"Facts:\n{_serialize_context_list(state['_context_list'])}\n\n"
                    f"{qa_prompt}\n\n"
                    f"Put your answer in <answer> tags."
                ),
            ),
        ]

    async def get_prompt_messages(self, state: vf.State) -> vf.Messages:
        if len(state["trajectory"]) == 0:
            if state.get("_skip_compaction"):
                return self._make_qa_prompt(state)
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
        content = (last_msg.content if hasattr(last_msg, "content") else "") or ""

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
            logger.info(f"[{state['_conv_id']}] Answer: {predicted[:80]} | compression={compression:.1f}%")
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

        # Cache
        conv_id = state["_conv_id"]
        _compaction_cache[conv_id] = list(state["_context_list"])
        _compaction_full_tokens[conv_id] = state.get("_full_history_tokens", 0)
        if conv_id in _compaction_events:
            _compaction_events[conv_id].set()

        logger.info(f"[{conv_id}] Compacted to {len(state['_context_list'])} items ({ctx_tokens} tokens)")

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
# RL mode: per-conversation rollouts (build → answer ALL QAs)
# ============================================================

class LoCoMoContextRLEnv(vf.MultiTurnEnv):
    """
    RL training env: one rollout per CONVERSATION, not per question.

    Each rollout:
      1. Process all sessions → build context_list (edit turns, get gradients)
      2. Answer sampled QAs for this conversation → compute avg F1 (QA turns, no gradients)
      3. Reward = avg F1 across sampled QAs

    No caching — each rollout builds independently so GRPO can compare strategies.
    """

    def __init__(self, num_qa_samples: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.num_qa_samples = num_qa_samples

    async def setup_state(self, state: vf.State) -> vf.State:
        state = await super().setup_state(state)
        # Disable thinking for Qwen3.5 models
        sa = state.get("sampling_args") or {}
        eb = sa.get("extra_body", {})
        eb["chat_template_kwargs"] = {"enable_thinking": False}
        sa["extra_body"] = eb
        state["sampling_args"] = sa
        logger.info(f"[RL] sampling_args.extra_body = {eb}")

        info = state.get("info", {})

        conversation = json.loads(info["conversation"])
        sessions = _get_sessions(conversation)
        all_qas = json.loads(info["all_qas"])

        # Sample QAs if configured (faster training with similar reward signal)
        import random
        if self.num_qa_samples and self.num_qa_samples < len(all_qas):
            all_qas = random.sample(all_qas, self.num_qa_samples)

        state["_sessions"] = sessions
        state["_session_idx"] = 1
        state["_context_list"] = []
        state["_edit_log"] = []
        state["_conv_id"] = info["conv_id"]

        # QA state
        state["_all_qas"] = all_qas
        state["_qa_idx"] = 0
        state["_qa_scores"] = []
        state["_sessions_done"] = False
        state["_question_sent"] = False

        # Compute full history tokens
        full_history = _format_all_sessions(sessions)
        state["_full_history_tokens"] = _count_tokens(full_history)

        logger.info(
            f"[{info['conv_id']}] RL rollout: {len(sessions)} sessions, "
            f"{len(all_qas)} QAs"
        )
        return state

    def _make_qa_prompt(self, state: vf.State) -> vf.Messages:
        qa = state["_all_qas"][state["_qa_idx"]]
        question = qa["question"]
        category = qa.get("category", 0)
        if category == 2:
            question = question + " Use DATE of CONVERSATION to answer with an approximate date."
        if category == 5:
            qa_prompt = QA_PROMPT_CAT_5.format(question)
        else:
            qa_prompt = QA_PROMPT.format(question)
        return [
            SystemMessage(role="system", content=QA_SYSTEM_PROMPT),
            UserMessage(
                role="user",
                content=(
                    f"Facts:\n{_serialize_context_list(state['_context_list'])}\n\n"
                    f"{qa_prompt}\n\n"
                    f"Put your answer in <answer> tags."
                ),
            ),
        ]

    async def get_prompt_messages(self, state: vf.State) -> vf.Messages:
        if len(state["trajectory"]) == 0:
            return state["prompt"]

        prev_prompt = state["trajectory"][-1]["prompt"]
        prev_completion = state["trajectory"][-1]["completion"]
        # Check if we have a skipped QA completion instead
        if state.get("_qa_completion"):
            prev_completion = state["_qa_completion"]
            state["_qa_completion"] = None

        messages = concat_messages([prev_prompt, prev_completion])
        env_resp = await self.env_response(messages, state)
        env_resp = maybe_normalize_messages(env_resp, field_name="env_response")

        if state.get("final_env_response") is not None:
            return concat_messages([messages, env_resp])

        if state["_sessions_done"]:
            return self._make_qa_prompt(state)
        else:
            return [
                SystemMessage(role="system", content=CONTEXT_SYSTEM_PROMPT),
                *env_resp,
            ]

    async def env_response(self, messages: vf.Messages, state: vf.State, **kwargs) -> vf.Messages:
        last_msg = messages[-1]
        content = (last_msg.content if hasattr(last_msg, "content") else "") or ""

        if state["_sessions_done"]:
            # Score QA answer
            predicted = _extract_answer_tag(content)
            qa = state["_all_qas"][state["_qa_idx"]]
            gold = str(qa.get("answer", ""))
            category = qa.get("category", 0)

            if category == 5:
                lower = predicted.lower()
                score = 1.0 if ("no information" in lower or "not mentioned" in lower) else 0.0
            else:
                score = _f1_score(predicted, gold)

            state["_qa_scores"].append(score)
            state["_qa_idx"] += 1

            # More QAs to answer?
            if state["_qa_idx"] < len(state["_all_qas"]):
                state["_question_sent"] = True
                return [UserMessage(role="user", content="[Next question]")]
            else:
                # All QAs done — compute final reward
                avg_f1 = sum(state["_qa_scores"]) / len(state["_qa_scores"])
                state["_avg_f1"] = avg_f1

                ctx_tokens = _count_tokens(_serialize_context_list(state["_context_list"]))
                full_tokens = state.get("_full_history_tokens", 0)
                compression = (ctx_tokens / full_tokens * 100) if full_tokens > 0 else 0
                state["_context_tokens"] = ctx_tokens
                state["_full_history_tokens"] = full_tokens

                logger.info(
                    f"[{state['_conv_id']}] RL rollout done: avg_f1={avg_f1:.4f} "
                    f"({len(state['_qa_scores'])} QAs) ctx={len(state['_context_list'])} items "
                    f"compression={compression:.1f}%"
                )

                final = [UserMessage(role="user", content="[Evaluation complete]")]
                state["final_env_response"] = final
                return final

        # Process edit from session turn
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
            "edit_ops": edit_ops,
            "context_list_len": len(state["_context_list"]),
            "context_tokens": ctx_tokens,
            "prefix_stability": ps,
        })

        sessions = state["_sessions"]

        if session_idx < len(sessions):
            _key, date, turns = sessions[session_idx]
            dialogue = _format_session(turns, date)
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
            # All sessions done → start QA phase
            state["_sessions_done"] = True
            state["_question_sent"] = True
            return [UserMessage(role="user", content="[Sessions complete, starting QA]")]

    async def add_trajectory_step(self, state: vf.State, trajectory_step):
        """Only edit turns get gradients. QA turns are for scoring only."""
        if state.get("_sessions_done"):
            state["_qa_completion"] = trajectory_step["completion"]
            return
        state["trajectory"].append(trajectory_step)

    async def render_completion(self, state: vf.State):
        completion = []
        for step in state["trajectory"]:
            completion.extend(step.get("completion", []))
        state["completion"] = completion
        dict.__setitem__(state, "_context_list", state.get("_context_list", []))
        dict.__setitem__(state, "_edit_log", state.get("_edit_log", []))
        dict.__setitem__(state, "_avg_f1", state.get("_avg_f1", 0.0))
        dict.__setitem__(state, "_qa_scores", state.get("_qa_scores", []))


# ============================================================
# Reward function
# ============================================================

def f1_reward(state: vf.State) -> float:
    answer = state.get("answer", "")
    info = state.get("info", {})
    category = info.get("category", 0) if isinstance(info, dict) else 0

    # Extract predicted answer from completion
    completion = state.get("completion", [])
    if completion:
        last = completion[-1]
        raw = last.content if hasattr(last, "content") else last.get("content", "")
        raw = raw.strip()
        # Always try to extract from <answer> tags first, then fall back to raw text
        predicted = _extract_answer_tag(raw)
    else:
        predicted = ""

    # Category 5 = unanswerable questions
    if category == 5:
        lower = predicted.lower()
        return 1.0 if ("no information" in lower or "not mentioned" in lower) else 0.0

    return _f1_score(predicted, answer)


def prefix_stability_metric(state: vf.State) -> float:
    """Average prefix stability across session steps (metric, not reward)."""
    edit_log = state.get("_edit_log", [])
    scores = [e.get("prefix_stability") for e in edit_log if "prefix_stability" in e]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def compression_metric(state: vf.State) -> float:
    """Compression ratio: context_tokens / full_history_tokens (metric, not reward)."""
    edit_log = state.get("_edit_log", [])
    answer_step = [e for e in edit_log if e.get("step") == "answer"]
    if answer_step:
        return float(answer_step[0].get("compression_pct", 0)) / 100.0
    return 0.0


def context_tokens_metric(state: vf.State) -> float:
    """Final context_list token count (metric, not reward)."""
    edit_log = state.get("_edit_log", [])
    answer_step = [e for e in edit_log if e.get("step") == "answer"]
    if answer_step:
        return float(answer_step[0].get("context_tokens", 0))
    return 0.0


def append_ratio_metric(state: vf.State) -> float:
    """Fraction of edits that were appends (vs pop/setitem). 1.0 = only appends."""
    edit_log = state.get("_edit_log", [])
    answer_step = [e for e in edit_log if e.get("step") == "answer"]
    if answer_step:
        return float(answer_step[0].get("append_ratio", 0))
    return 0.0


def total_edits_metric(state: vf.State) -> float:
    """Total number of edit operations across all session steps."""
    edit_log = state.get("_edit_log", [])
    answer_step = [e for e in edit_log if e.get("step") == "answer"]
    if answer_step:
        return float(answer_step[0].get("total_edits", 0))
    return 0.0


def max_list_length_metric(state: vf.State) -> float:
    """Peak context_list length during the run."""
    edit_log = state.get("_edit_log", [])
    answer_step = [e for e in edit_log if e.get("step") == "answer"]
    if answer_step:
        return float(answer_step[0].get("max_list_length", 0))
    return 0.0


def rl_reward(state: vf.State) -> float:
    """Reward for RL mode: avg F1 across all QAs (computed inside the env)."""
    return float(state.get("_avg_f1", 0.0))


# ============================================================
# Dataset builder
# ============================================================

def _build_dataset(mode: str, dataset_name: str, split: str, n: int, conv_id: str | None = None, prompt_version: str = "v1"):
    def builder():
        ds = load_dataset(dataset_name, split=split)
        if conv_id:
            ds = ds.filter(lambda row: row["conv_id"] == conv_id)
        if n > 0:
            ds = ds.select(range(min(n, len(ds))))

        if mode == "baseline":

            def format_baseline(row):
                conversation = json.loads(row["conversation"])
                sessions = _get_sessions(conversation)
                history = _format_all_sessions(sessions)
                question = row["question"]
                category = row["category"]

                # Per-category question formatting (matches original LoCoMo)
                if category == 2:
                    question = question + " Use DATE of CONVERSATION to answer with an approximate date."

                if category == 5:
                    qa_prompt = QA_PROMPT_CAT_5.format(question)
                else:
                    qa_prompt = QA_PROMPT.format(question)

                return {
                    "prompt": [
                        {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                f"Below is a conversation between {row['speaker_a']} and {row['speaker_b']}.\n\n"
                                f"{history}\n\n"
                                f"{qa_prompt}"
                            ),
                        },
                    ],
                    "answer": row["answer"],
                    "info": json.dumps(
                        {
                            "category": row["category"],
                            "conv_id": row["conv_id"],
                        }
                    ),
                }

            ds = ds.map(format_baseline)

        elif mode == "compaction":

            def format_compaction(row):
                conversation = json.loads(row["conversation"])
                sessions = _get_sessions(conversation)
                history = _format_all_sessions(sessions)
                return {
                    "prompt": [
                        {"role": "system", "content": COMPACTION_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                f"Below is a conversation between {row['speaker_a']} and {row['speaker_b']}.\n\n"
                                f"{history}\n\n"
                                f"Extract all important facts into context_list using <edit> tags."
                            ),
                        },
                    ],
                    "answer": row["answer"],
                    "info": json.dumps(
                        {
                            "conversation": row["conversation"],
                            "question": row["question"],
                            "category": row["category"],
                            "conv_id": row["conv_id"],
                        }
                    ),
                }

            ds = ds.map(format_compaction)

        elif mode == "context_rl":
            # One row per CONVERSATION (not per question)
            # Bundle all QAs into info for the RL env
            from collections import defaultdict
            conv_qas = defaultdict(list)
            conv_data = {}
            for row in ds:
                cid = row["conv_id"]
                conv_qas[cid].append({
                    "question": row["question"],
                    "answer": row["answer"],
                    "category": row["category"],
                })
                if cid not in conv_data:
                    conv_data[cid] = row

            selected_prompt = PROMPT_VERSIONS.get(prompt_version, CONTEXT_SYSTEM_PROMPT)
            rows = []
            for cid in sorted(conv_data.keys()):
                row = conv_data[cid]
                conversation = json.loads(row["conversation"])
                sessions = _get_sessions(conversation)
                first_session = _format_session(sessions[0][2], sessions[0][1])
                rows.append({
                    "prompt": [
                        {"role": "system", "content": selected_prompt},
                        {
                            "role": "user",
                            "content": (
                                f"Current context list: []\n\n"
                                f"New dialogue:\n{first_session}"
                            ),
                        },
                    ],
                    "answer": "",
                    "info": json.dumps({
                        "conversation": row["conversation"],
                        "all_qas": json.dumps(conv_qas[cid]),
                        "conv_id": cid,
                    }),
                })
            return Dataset.from_list(rows)

        else:  # context mode

            def format_context(row):
                conversation = json.loads(row["conversation"])
                sessions = _get_sessions(conversation)
                first_session = _format_session(sessions[0][2], sessions[0][1])
                return {
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
                    "answer": row["answer"],
                    "info": json.dumps(
                        {
                            "conversation": row["conversation"],
                            "question": row["question"],
                            "category": row["category"],
                            "conv_id": row["conv_id"],
                        }
                    ),
                }

            ds = ds.map(format_context)

        return ds

    return builder


# ============================================================
# Entry point
# ============================================================

def load_environment(
    mode: str = "context",
    dataset_name: str = "bhoy/locomo",
    dataset_split: str = "test",
    num_examples: int = -1,
    conv_id: str = "",
    prompt_version: str = "v1",
    num_qa_samples: int = -1,
) -> vf.Environment:
    """
    Load the LoCoMo context management environment.

    Args:
        mode: "baseline", "context", "compaction", or "context_rl"
        dataset_name: HuggingFace dataset path
        dataset_split: Dataset split to use
        num_examples: Limit number of examples (-1 for all)
        conv_id: Filter to a single conversation (e.g. "conv-26"). Empty string for all.
        prompt_version: "v1" (aggressive), "v3" (softer), or "minimal" (bare bones). For context/context_rl modes.
        num_qa_samples: Number of QAs to sample per conversation in context_rl mode (-1 for all).
    """
    # Clear cache between runs
    _context_cache.clear()
    _cache_edit_logs.clear()
    _cache_events.clear()
    _cache_full_history_tokens.clear()

    if mode == "context_rl":
        rubric = vf.Rubric(funcs=[rl_reward])
        rubric.add_metric(prefix_stability_metric, weight=0.0)
        rubric.add_metric(context_tokens_metric, weight=0.0)
        rubric.add_metric(append_ratio_metric, weight=0.0)
    else:
        rubric = vf.Rubric(funcs=[f1_reward])
    # Add context-mode-only metrics (weight=0 so they don't affect reward)
    if mode in ("context", "compaction"):
        rubric.add_metric(prefix_stability_metric, weight=0.0)
        rubric.add_metric(compression_metric, weight=0.0)
        rubric.add_metric(context_tokens_metric, weight=0.0)
        rubric.add_metric(append_ratio_metric, weight=0.0)
        rubric.add_metric(total_edits_metric, weight=0.0)
        rubric.add_metric(max_list_length_metric, weight=0.0)
    dataset_builder = _build_dataset(mode, dataset_name, dataset_split, num_examples, conv_id or None, prompt_version=prompt_version)

    if mode == "baseline":
        return vf.SingleTurnEnv(
            dataset=dataset_builder,
            rubric=rubric,
        )
    elif mode == "context":
        return LoCoMoContextEnv(
            dataset=dataset_builder,
            rubric=rubric,
            max_turns=50,
        )
    elif mode == "compaction":
        _compaction_cache.clear()
        _compaction_events.clear()
        _compaction_full_tokens.clear()
        return LoCoMoCompactionEnv(
            dataset=dataset_builder,
            rubric=rubric,
            max_turns=5,
        )
    elif mode == "context_rl":
        qa_samples = num_qa_samples if num_qa_samples > 0 else None
        return LoCoMoContextRLEnv(
            dataset=dataset_builder,
            rubric=rubric,
            max_turns=250,
            num_qa_samples=qa_samples,
        )
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'baseline', 'context', 'compaction', or 'context_rl'.")
