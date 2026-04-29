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

from datasets import load_dataset

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
    return match.group(1).strip() if match else None


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
        content = last_msg.content if hasattr(last_msg, "content") else ""

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

    async def render_completion(self, state: vf.State):
        """Override to capture just the QA response as completion."""
        if state["trajectory"]:
            last = state["trajectory"][-1]
            state["completion"] = list(last["completion"])
        else:
            state["completion"] = []


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


# ============================================================
# Dataset builder
# ============================================================

def _interleave_by_conv(ds):
    """
    Reorder dataset so rows from different conversations interleave.
    Result: row 0 = conv-A q0, row 1 = conv-B q0, ... row N = conv-A q1, row N+1 = conv-B q1, ...
    This ensures the first N rows cover N different conversations → N builders fire in parallel.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for i, conv in enumerate(ds["conv_id"]):
        groups[conv].append(i)
    # Round-robin: take one index from each group until all exhausted
    interleaved_indices = []
    max_len = max(len(v) for v in groups.values())
    for i in range(max_len):
        for conv_id in groups:
            if i < len(groups[conv_id]):
                interleaved_indices.append(groups[conv_id][i])
    return ds.select(interleaved_indices)


def _build_dataset(mode: str, dataset_name: str, split: str, n: int,
                   conv_id: str | None = None, interleave: bool = True):
    def builder():
        ds = load_dataset(dataset_name, split=split)
        if conv_id:
            ds = ds.filter(lambda row: row["conv_id"] == conv_id)
        # Interleave BEFORE limiting so the limit is applied after the reorder
        if interleave and not conv_id:
            ds = _interleave_by_conv(ds)
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
    interleave: bool = True,
) -> vf.Environment:
    """
    Load the LoCoMo context management environment.

    Args:
        mode: "baseline" (full history per question) or "context" (incremental fact management)
        dataset_name: HuggingFace dataset path
        dataset_split: Dataset split to use
        num_examples: Limit number of examples (-1 for all)
        conv_id: Filter to a single conversation (e.g. "conv-26"). Empty string for all.
        interleave: If True, reorder rows so first N rows cover N different conversations.
                    Lets multiple "builder" rollouts run in parallel when max_concurrent is limited.
                    Ignored if conv_id is set (only one conv to process).
    """
    # Clear cache between runs
    _context_cache.clear()
    _cache_edit_logs.clear()
    _cache_events.clear()

    rubric = vf.Rubric(funcs=[f1_reward])
    # Add context-mode-only metrics (weight=0 so they don't affect reward)
    if mode == "context":
        rubric.add_metric(prefix_stability_metric, weight=0.0)
        rubric.add_metric(compression_metric, weight=0.0)
        rubric.add_metric(context_tokens_metric, weight=0.0)
        rubric.add_metric(append_ratio_metric, weight=0.0)
        rubric.add_metric(total_edits_metric, weight=0.0)
        rubric.add_metric(max_list_length_metric, weight=0.0)
    dataset_builder = _build_dataset(
        mode, dataset_name, dataset_split, num_examples,
        conv_id or None, interleave=interleave,
    )

    if mode == "baseline":
        return vf.SingleTurnEnv(
            dataset=dataset_builder,
            rubric=rubric,
        )
    elif mode == "context":
        return LoCoMoContextEnv(
            dataset=dataset_builder,
            rubric=rubric,
            max_turns=50,  # up to 32 sessions + 1 QA turn + buffer
        )
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'baseline' or 'context'.")
