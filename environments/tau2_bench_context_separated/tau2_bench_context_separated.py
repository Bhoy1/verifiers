"""
τ²-bench with SEPARATED context-managed agent.

Each logical turn splits into 2 model calls:
  Call 1 (edit): Model sees context_list + latest message → outputs <edit> block only
  Call 2 (respond): Model sees UPDATED context_list + latest message → outputs text/tool call

Only Call 1 (edit) is added to the trajectory for RL training.
Call 2 (respond) is used for tau2 orchestration but skipped from gradient computation.
"""

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any, Callable, TypeVar, cast

from typing_extensions import TypedDict

T = TypeVar("T")

import verifiers as vf
from datasets import Dataset
from loguru import logger as loguru_logger
from verifiers.types import SystemMessage, UserMessage
from verifiers.utils.message_utils import concat_messages, maybe_normalize_messages

loguru_logger.remove()
pylogger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message="Pydantic serializer warnings", category=UserWarning)

from tau2.agent.llm_agent import (
    LLMAgent,
    LLMAgentState,
    is_valid_agent_history_message,
)
from tau2.config import (
    DEFAULT_LLM_ARGS_AGENT,
    DEFAULT_MAX_ERRORS,
    DEFAULT_MAX_STEPS,
)
# Override NL assertions judge model and fix temperature for gpt-5.5
import tau2.config as _tau2_config
import tau2.evaluator.evaluator_nl_assertions as _nl_eval
_tau2_config.DEFAULT_LLM_NL_ASSERTIONS = "gpt-5.5"
_nl_eval.DEFAULT_LLM_NL_ASSERTIONS = "gpt-5.5"
# gpt-5.5 doesn't support temperature=0, override with empty args
_tau2_config.DEFAULT_LLM_ARGS_USER = {}
_tau2_config.DEFAULT_LLM_NL_ASSERTIONS_ARGS = {}
_nl_eval.DEFAULT_LLM_NL_ASSERTIONS_ARGS = {}
DEFAULT_LLM_ARGS_USER = {}  # local override (imported by value above would be stale)

# Fix Windows encoding
import tau2.utils.io_utils as _tau2_io
_original_load_file = _tau2_io.load_file
def _load_file_utf8(path, **kwargs):
    path_obj = Path(path)
    if path_obj.suffix in (".txt", ".md") and "encoding" not in kwargs:
        kwargs["encoding"] = "utf-8"
    return _original_load_file(path, **kwargs)
_tau2_io.load_file = _load_file_utf8
try:
    import tau2.domains.telecom.environment as _telecom_env
    _telecom_env.load_file = _load_file_utf8
except Exception:
    pass

from tau2.data_model.message import (
    AssistantMessage as Tau2AssistantMessage,
    Message,
    MultiToolMessage,
    ToolCall,
    ToolMessage as Tau2ToolMessage,
    UserMessage as Tau2UserMessage,
)
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE, Role
from tau2.registry import registry
from tau2.run import load_tasks
from tau2.user.user_simulator import UserSimulator, UserState, is_valid_user_history_message
from tau2.utils.utils import DATA_DIR, format_time, get_now
from verifiers.envs.multiturn_env import MultiTurnEnv

# ============================================================
# Shared helpers (inlined from tau2_bench_context for standalone use)
# ============================================================

# -- Tokenizer --
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

# -- RestrictedPython --
try:
    from RestrictedPython import compile_restricted_exec, limited_builtins, utility_builtins
    from RestrictedPython.Guards import guarded_iter_unpack_sequence, safer_getattr
    from RestrictedPython.PrintCollector import PrintCollector
    _HAS_RESTRICTED_PYTHON = True
except ImportError:
    _HAS_RESTRICTED_PYTHON = False

# -- Constants --
DEFAULT_USER_MODEL = "gpt-5.5"
DEFAULT_USER_BASE_URL = "https://api.openai.com/v1"
DEFAULT_USER_API_KEY_VAR = "OPENAI_API_KEY"
DEFAULT_MAX_WORKERS = 128


def _serialize_context_list(context_list: list[str]) -> str:
    if not context_list:
        return "[]"
    items = ", ".join(f'"{item}"' for item in context_list)
    return f"[{items}]"


def _extract_edit_code(response: str | None) -> str | None:
    if not response:
        return None
    match = re.search(r"<edit>(.*?)</edit>", response, re.DOTALL)
    return match.group(1).strip() if match else None


def _strip_edit_block(response: str | None) -> str:
    if not response:
        return ""
    text = re.sub(r"<edit>.*?</edit>", "", response, flags=re.DOTALL).strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
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


def download_tau2_data():
    if os.path.exists(DATA_DIR) and os.path.exists(DATA_DIR / "tau2" / "domains"):
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="tau2_bench_"))
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/sierra-research/tau2-bench.git", str(temp_dir)],
            check=True, capture_output=True,
        )
        src_data = temp_dir / "data"
        if os.path.exists(src_data):
            shutil.copytree(src_data, DATA_DIR, dirs_exist_ok=True)
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to download tau2-bench data: {e}")
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def tau_msgs_to_vf_msgs(tau_msgs: list[Message]) -> vf.Messages:
    def tau_msg_to_vf_msg(tau_msg: Message) -> vf.Message:
        if isinstance(tau_msg, Tau2AssistantMessage):
            if tau_msg.tool_calls:
                return vf.AssistantMessage(
                    content=tau_msg.content,
                    tool_calls=[
                        vf.ToolCall(id=tc.id, name=tc.name, arguments=json.dumps(tc.arguments))
                        for tc in tau_msg.tool_calls
                    ],
                )
            return vf.AssistantMessage(content=tau_msg.content)
        elif isinstance(tau_msg, Tau2UserMessage):
            return vf.UserMessage(content=tau_msg.content or "")
        elif isinstance(tau_msg, Tau2ToolMessage):
            return vf.ToolMessage(tool_call_id=tau_msg.id, content=tau_msg.content or "")
        else:
            raise ValueError(f"Unknown message type: {type(tau_msg)}")
    return [tau_msg_to_vf_msg(m) for m in tau_msgs]


def _format_latest_messages(messages: vf.Messages) -> str:
    parts = []
    for msg in messages:
        if hasattr(msg, "role"):
            role = msg.role
        else:
            role = msg.get("role", "unknown")
        content = msg.content if hasattr(msg, "content") else msg.get("content", "")
        content = content or ""
        if role == "user":
            parts.append(f"Customer: {content}")
        elif role == "tool":
            parts.append(f"Tool result:\n{content}")
        else:
            parts.append(str(content))
    return "\n\n".join(parts)


class Tau2BenchState(TypedDict):
    task: Task
    agent: LLMAgent
    agent_state: LLMAgentState
    user: UserSimulator
    user_state: UserState
    environment: Environment
    trajectory: list[Message]
    message: Message
    from_role: Role
    to_role: Role
    done: bool
    termination_reason: TerminationReason | None
    step_count: int
    num_errors: int


class Tau2ContextMonitorRubric(vf.Rubric):
    def __init__(self):
        super().__init__()
        self.add_metric(self.num_errors)
        self.add_metric(self.num_steps)
        self.add_metric(self.num_assistant_tool_calls)
        self.add_metric(self.num_user_tool_calls)

    def num_errors(self, state: vf.State) -> float:
        tau2 = cast(Tau2BenchState, state["tau2"])
        return tau2["num_errors"]

    def num_steps(self, state: vf.State) -> float:
        tau2 = cast(Tau2BenchState, state["tau2"])
        return tau2["step_count"]

    def num_assistant_tool_calls(self, state: vf.State) -> float:
        return state.get("num_assistant_tool_calls", 0.0)

    def num_user_tool_calls(self, state: vf.State) -> float:
        return state.get("num_user_tool_calls", 0.0)


def prefix_stability_metric(state: vf.State) -> float:
    edit_log = state.get("_edit_log", [])
    scores = [e.get("prefix_stability") for e in edit_log if "prefix_stability" in e]
    return sum(scores) / len(scores) if scores else 0.0


def append_ratio_metric(state: vf.State) -> float:
    edit_log = state.get("_edit_log", [])
    all_ops = []
    for e in edit_log:
        all_ops.extend(e.get("edit_ops", []))
    if not all_ops:
        return 0.0
    return sum(1 for op in all_ops if op["type"] == "append") / len(all_ops)


def context_tokens_metric(state: vf.State) -> float:
    ctx = state.get("_context_list", [])
    return float(_count_tokens(_serialize_context_list(ctx)))


def total_edits_metric(state: vf.State) -> float:
    edit_log = state.get("_edit_log", [])
    return float(sum(len(e.get("edit_ops", [])) for e in edit_log))


def max_list_length_metric(state: vf.State) -> float:
    edit_log = state.get("_edit_log", [])
    return float(max((e.get("context_list_len", 0) for e in edit_log), default=0))


# ============================================================
# Separated system prompts — one for edit, one for respond
# ============================================================

EDIT_SYSTEM_PROMPT = """You manage a compressed context list for a customer service conversation.
You see your current context list and the latest message.

Save important information from the latest message using <edit> tags.

<edit>
context_list.append("new fact")
context_list[0] = "updated fact"
context_list.pop(0)
</edit>

Rules:
- Save exact IDs, codes, values, and names verbatim (never paraphrase)
- Save user preferences and decisions
- Copy names, IDs, and reservation codes EXACTLY
- Output ONLY the <edit> block, nothing else"""

RESPOND_SYSTEM_PROMPT = """You are a customer service agent that helps the user according to the <policy> provided below.

You see a compressed context list (your memory) and the latest message.
Based on these, either send a message to the user OR make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy."""

RESPOND_SYSTEM_PROMPT_FULL = """<instructions>
{respond_instruction}
</instructions>
<policy>
{domain_policy}
</policy>"""


# ============================================================
# Separated environment
# ============================================================

class Tau2BenchSeparatedEnv(MultiTurnEnv):
    """
    τ²-bench with SEPARATED context management.

    Each logical turn = 2 model calls:
      Call 1 (edit): [edit_system, ctx + latest] → model outputs <edit> block only
        → ADDED to trajectory (gets gradients during RL)
      Call 2 (respond): [respond_system+policy, updated_ctx + latest] → model outputs text/tool
        → SKIPPED from trajectory (no gradients)
        → Goes through tau2 orchestration (tool exec, user sim)
    """

    def __init__(
        self,
        domain: str,
        user_model: str = DEFAULT_USER_MODEL,
        user_args: dict = DEFAULT_LLM_ARGS_USER,
        user_base_url: str = DEFAULT_USER_BASE_URL,
        user_api_key_var: str = DEFAULT_USER_API_KEY_VAR,
        max_steps: int = DEFAULT_MAX_STEPS,
        max_errors: int = DEFAULT_MAX_ERRORS,
        max_workers: int = DEFAULT_MAX_WORKERS,
        max_turns: int = -1,
        task_ids: list[int] | None = None,
        **kwargs,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tau2-sep")
        self.domain = domain
        self.user_model = user_model
        self.user_args = {**user_args, "api_base": user_base_url, "api_key": os.getenv(user_api_key_var)}
        self.max_steps = max_steps
        self.max_errors = max_errors
        self.task_ids = set(task_ids) if task_ids else None

        eval_dataset, tool_defs = self._create_dataset(domain)
        rubric = self._create_rubric(domain)

        # Double max_turns since each logical turn = 2 model calls
        if max_turns > 0:
            max_turns = max_turns * 2

        super().__init__(
            dataset=eval_dataset,
            eval_dataset=eval_dataset,
            rubric=rubric,
            tool_defs=tool_defs,
            max_turns=max_turns,
            **kwargs,
        )
        self.add_rubric(Tau2ContextMonitorRubric())

    async def _run_in_thread(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.thread_pool, partial(func, *args, **kwargs))

    def _create_dataset(self, domain: str) -> tuple[Dataset, list[vf.Tool]]:
        EnvironmentConstructor = registry.get_env_constructor(domain)
        environment = EnvironmentConstructor()
        tools = environment.get_tools()
        oai_tools = [tool.openai_schema for tool in tools] if tools else []
        tool_defs = [
            vf.Tool(
                name=tool["function"]["name"],
                description=tool["function"]["description"],
                parameters=tool["function"]["parameters"],
                strict=False,
            )
            for tool in oai_tools
        ] if oai_tools else []

        # Edit prompt doesn't need policy — respond prompt does
        edit_system = EDIT_SYSTEM_PROMPT
        respond_system = RESPOND_SYSTEM_PROMPT_FULL.format(
            respond_instruction=RESPOND_SYSTEM_PROMPT,
            domain_policy=environment.policy,
        )

        def process_task(task: Task) -> dict:
            return {
                "prompt": [{"role": "system", "content": edit_system}],
                "info": task.model_dump_json(exclude_none=True),
            }

        tasks = load_tasks(task_set_name=domain, task_split_name="base")
        if self.task_ids is not None:
            tasks = [t for i, t in enumerate(tasks) if i in self.task_ids]
        rows = [process_task(task) for task in tasks]
        dataset = Dataset.from_list(rows)

        # Store respond system prompt for Call 2
        self._respond_system = respond_system

        return dataset, tool_defs

    def _create_rubric(self, domain: str) -> vf.Rubric:
        async def evaluate_tau2_task(state, **kwargs) -> float:
            tau2 = cast(Tau2BenchState, state["tau2"])
            simulation = SimulationRun(
                id=f"{domain}_{tau2['task'].id}_{datetime.now().isoformat()}",
                task_id=tau2["task"].id,
                messages=tau2["trajectory"],
                termination_reason=tau2["termination_reason"] or TerminationReason.AGENT_ERROR,
                timestamp=datetime.now().isoformat(),
                start_time=datetime.now().isoformat(),
                end_time=datetime.now().isoformat(),
                duration=0.0,
                agent_cost=0.0,
                user_cost=0.0,
            )
            reward_info = evaluate_simulation(
                simulation=simulation,
                task=tau2["task"],
                evaluation_type=EvaluationType.ALL_WITH_NL_ASSERTIONS,
                solo_mode=False,
                domain=domain,
            )
            return reward_info.reward
        return vf.Rubric(funcs=[evaluate_tau2_task], weights=[1.0])

    # ---- Stop conditions ----
    @vf.stop
    async def max_steps_reached(self, state: vf.State) -> bool:
        tau2 = cast(Tau2BenchState, state["tau2"])
        return tau2["done"] and tau2["termination_reason"] == TerminationReason.MAX_STEPS

    @vf.stop
    async def too_many_errors(self, state: vf.State) -> bool:
        tau2 = cast(Tau2BenchState, state["tau2"])
        return tau2["done"] and tau2["termination_reason"] == TerminationReason.TOO_MANY_ERRORS

    @vf.stop
    async def user_stopped(self, state: vf.State) -> bool:
        tau2 = cast(Tau2BenchState, state["tau2"])
        return tau2["done"] and tau2["termination_reason"] == TerminationReason.USER_STOP

    @vf.stop
    async def agent_stopped(self, state: vf.State) -> bool:
        tau2 = cast(Tau2BenchState, state["tau2"])
        return tau2["done"] and tau2["termination_reason"] == TerminationReason.AGENT_STOP

    # ---- State setup ----
    async def setup_state(self, state: vf.State) -> vf.State:
        state["sampling_args"] = {**DEFAULT_LLM_ARGS_AGENT, **(state["sampling_args"] or {})}
        # Disable thinking for RL training
        eb = state["sampling_args"].setdefault("extra_body", {})
        eb["chat_template_kwargs"] = {"enable_thinking": False}
        await self._initialize(state)

        state["_context_list"] = []
        state["_edit_log"] = []
        state["_latest_message"] = ""
        state["_awaiting_response"] = False  # False = next call is edit, True = next call is respond

        # Step until first agent turn
        setup_messages = []
        tau2 = cast(Tau2BenchState, state["tau2"])
        while not (tau2["done"] or tau2["to_role"] == Role.AGENT):
            new_messages = await self._step(state["prompt"] + setup_messages, state)
            if tau2["step_count"] >= self.max_steps:
                tau2["done"] = True
                tau2["termination_reason"] = TerminationReason.MAX_STEPS
            if tau2["num_errors"] >= self.max_errors:
                tau2["done"] = True
                tau2["termination_reason"] = TerminationReason.TOO_MANY_ERRORS
            setup_messages.extend(new_messages)

        if setup_messages:
            state["_latest_message"] = _format_latest_messages(setup_messages)

        # Initial prompt is the edit prompt
        state["prompt"] = [
            SystemMessage(role="system", content=EDIT_SYSTEM_PROMPT),
            UserMessage(
                role="user",
                content=(
                    f"Current context list: {_serialize_context_list(state['_context_list'])}\n\n"
                    f"Latest message:\n{state['_latest_message']}"
                ),
            ),
        ]
        return state

    # ---- Prompt construction ----
    async def get_prompt_messages(self, state: vf.State) -> vf.Messages:
        if len(state["trajectory"]) == 0 and not state.get("_response_trajectories"):
            return state["prompt"]

        # Get previous completion
        if state.get("_last_response_completion"):
            # After Call 2 (respond) — process through tau2, build next edit prompt
            prev_completion = state["_last_response_completion"]
            state["_last_response_completion"] = None

            # Build fake messages for env_response
            messages = [UserMessage(role="user", content="")] + list(prev_completion)
            env_resp = await self._process_response(messages, state)
            env_resp = maybe_normalize_messages(env_resp, field_name="env_response")

            tau2 = cast(Tau2BenchState, state["tau2"])
            if tau2["done"]:
                state["final_env_response"] = env_resp
                return concat_messages([messages, env_resp])

            # Build EDIT prompt (Call 1) for next turn
            latest = _format_latest_messages(env_resp) if env_resp else state.get("_latest_message", "")
            state["_latest_message"] = latest
            state["_awaiting_response"] = False

            return [
                SystemMessage(role="system", content=EDIT_SYSTEM_PROMPT),
                UserMessage(
                    role="user",
                    content=(
                        f"Current context list: {_serialize_context_list(state['_context_list'])}\n\n"
                        f"Latest message:\n{latest}"
                    ),
                ),
            ]

        elif state["trajectory"]:
            # After Call 1 (edit) — execute edit, build respond prompt
            prev_completion = state["trajectory"][-1]["completion"]
            content = prev_completion[-1].content if prev_completion and hasattr(prev_completion[-1], "content") else ""
            content = content if isinstance(content, str) else ""

            # Execute edit
            before = list(state["_context_list"])
            edit_code = _extract_edit_code(content)
            edit_success = True
            edit_error = ""
            if edit_code:
                edit_success, edit_error = _execute_context_edit(edit_code, state["_context_list"])
                if not edit_success:
                    state["_context_list"] = list(before)

            # Log edit
            ps = _prefix_stability(before, state["_context_list"])
            edit_ops = _parse_edit_operations(edit_code) if edit_success else []
            ctx_tokens = _count_tokens(_serialize_context_list(state["_context_list"]))
            state["_edit_log"].append({
                "step": len(state["_edit_log"]),
                "edit_code": edit_code,
                "edit_success": edit_success,
                "edit_error": edit_error if not edit_success else "",
                "edit_ops": edit_ops,
                "context_list_len": len(state["_context_list"]),
                "context_tokens": ctx_tokens,
                "prefix_stability": ps,
            })

            # Build RESPOND prompt (Call 2) with UPDATED context
            state["_awaiting_response"] = True
            return [
                SystemMessage(role="system", content=self._respond_system),
                UserMessage(
                    role="user",
                    content=(
                        f"Current context list: {_serialize_context_list(state['_context_list'])}\n\n"
                        f"Latest message:\n{state['_latest_message']}"
                    ),
                ),
            ]

        return state["prompt"]

    # ---- env_response (required by MultiTurnEnv, logic handled in get_prompt_messages) ----
    async def env_response(self, messages: vf.Messages, state: vf.State, **kwargs) -> vf.Messages:
        # In the separated architecture, env_response logic is handled in get_prompt_messages
        # after Call 2 (respond) completes. This stub satisfies the abstract method requirement.
        return []

    # ---- Trajectory control ----
    async def add_trajectory_step(self, state: vf.State, trajectory_step):
        if state.get("_awaiting_response"):
            # Call 2 (respond) — skip from trajectory, save for processing
            state["_last_response_completion"] = trajectory_step["completion"]
            if not hasattr(state, "_response_trajectories"):
                state["_response_trajectories"] = []
            state["_response_trajectories"] = state.get("_response_trajectories", [])
            state["_response_trajectories"].append(trajectory_step)
            return
        # Call 1 (edit) — add to trajectory (gets gradients)
        state["trajectory"].append(trajectory_step)

    # ---- Process Call 2 response through tau2 ----
    async def _process_response(self, messages, state):
        """Process the respond call's output through tau2 orchestration."""
        tau2 = cast(Tau2BenchState, state["tau2"])
        last_message = messages[-1]
        content = last_message.content if hasattr(last_message, "content") else ""
        content = content if isinstance(content, str) and content else None

        tool_calls = getattr(last_message, "tool_calls", None) or []
        if isinstance(tool_calls, list) and len(tool_calls) > 128:
            tool_calls = tool_calls[:128]
        state["num_assistant_tool_calls"] = state.get("num_assistant_tool_calls", 0) + len(tool_calls)

        tau2_tool_calls = []
        for tc in tool_calls:
            try:
                arguments = json.loads(tc.arguments)
            except json.JSONDecodeError:
                continue
            tau2_tool_calls.append(
                ToolCall(id=tc.id, name=tc.name, arguments=arguments, requestor="assistant")
            )
        tau2_tool_calls = tau2_tool_calls or None

        response = state.get("_response_trajectories", [{}])[-1].get("response") if state.get("_response_trajectories") else None
        raw_data = response.to_dict() if response and hasattr(response, "to_dict") else None
        tau2_asst_msg = Tau2AssistantMessage(
            role="assistant",
            content=content,
            tool_calls=tau2_tool_calls,
            raw_data=raw_data,
        )

        tau2["agent_state"].messages.append(tau2_asst_msg)
        try:
            tau2_asst_msg.validate()
        except ValueError:
            tau2["done"] = True
            tau2["termination_reason"] = TerminationReason.AGENT_ERROR
            tau2["trajectory"].append(tau2_asst_msg)
            return []

        if tau2["agent"].is_stop(tau2_asst_msg):
            tau2["done"] = True
            tau2["termination_reason"] = TerminationReason.AGENT_STOP

        tau2["trajectory"].append(tau2_asst_msg)
        tau2["message"] = tau2_asst_msg
        tau2["from_role"] = Role.AGENT
        tau2["to_role"] = Role.ENV if tau2_tool_calls else Role.USER
        tau2["step_count"] += 1
        tau2["environment"].sync_tools()

        response_messages = []
        while not (tau2["done"] or tau2["to_role"] == Role.AGENT):
            new_messages = await self._step(messages + response_messages, state)
            if tau2["step_count"] >= self.max_steps and tau2["to_role"] != Role.ENV:
                tau2["done"] = True
                tau2["termination_reason"] = TerminationReason.MAX_STEPS
            if tau2["num_errors"] >= self.max_errors:
                tau2["done"] = True
                tau2["termination_reason"] = TerminationReason.TOO_MANY_ERRORS
            response_messages.extend(new_messages)

        return response_messages

    async def render_completion(self, state: vf.State):
        completion = []
        for step in state["trajectory"]:
            completion.extend(step.get("completion", []))
        # Include response completions for full conversation view
        for step in state.get("_response_trajectories", []):
            completion.extend(step.get("completion", []))
        state["completion"] = completion
        dict.__setitem__(state, "_context_list", state.get("_context_list", []))
        dict.__setitem__(state, "_edit_log", state.get("_edit_log", []))

    # ---- Initialize + Step (same as original tau2_bench_context) ----
    async def _initialize(self, state: vf.State):
        global registry
        task = Task.model_validate(state["info"])
        EnvironmentConstructor = registry.get_env_constructor(self.domain)
        environment = await self._run_in_thread(EnvironmentConstructor)
        agent = LLMAgent(
            tools=environment.get_tools(),
            domain_policy=environment.get_policy(),
            llm=state["model"],
            llm_args=state["sampling_args"],
        )
        try:
            user_tools = environment.get_user_tools()
        except Exception:
            user_tools = None
        user = UserSimulator(
            tools=user_tools,
            instructions=str(task.user_scenario),
            llm=self.user_model,
            llm_args=self.user_args,
        )
        initial_state = task.initial_state
        initialization_data = initial_state.initialization_data if initial_state else None
        initialization_actions = initial_state.initialization_actions if initial_state else None
        message_history = deepcopy(initial_state.message_history) if initial_state and initial_state.message_history else []
        for msg in message_history:
            msg.turn_idx = None
        message_history = self._add_timestamps(message_history)
        environment.set_state(
            initialization_data=initialization_data,
            initialization_actions=initialization_actions,
            message_history=message_history,
        )
        done = False
        termination_reason = None
        if message_history:
            last_message = message_history[-1]
            if isinstance(last_message, Tau2AssistantMessage):
                from_role = Role.AGENT
                to_role = Role.USER if not last_message.is_tool_call() else Role.ENV
                agent_state = agent.get_init_state(message_history=[m for m in message_history if is_valid_agent_history_message(m)])
                user_state = user.get_init_state(message_history=[m for m in message_history[:-1] if is_valid_user_history_message(m)])
                if agent.is_stop(last_message):
                    done = True
                    termination_reason = TerminationReason.AGENT_STOP
            elif isinstance(last_message, Tau2UserMessage):
                from_role = Role.USER
                to_role = Role.AGENT if not last_message.is_tool_call() else Role.ENV
                user_state = user.get_init_state(message_history=[m for m in message_history if is_valid_user_history_message(m)])
                agent_state = agent.get_init_state(message_history=[m for m in message_history[:-1] if is_valid_agent_history_message(m)])
                done = UserSimulator.is_stop(last_message)
                if done:
                    termination_reason = TerminationReason.USER_STOP
            elif isinstance(last_message, Tau2ToolMessage):
                from_role = Role.ENV
                if last_message.requestor == "assistant":
                    to_role = Role.AGENT
                    agent_state = agent.get_init_state(message_history=[m for m in message_history[:-1] if is_valid_agent_history_message(m)])
                    user_state = user.get_init_state(message_history=[m for m in message_history if is_valid_user_history_message(m)])
                else:
                    to_role = Role.USER
                    agent_state = agent.get_init_state(message_history=[m for m in message_history if is_valid_agent_history_message(m)])
                    user_state = user.get_init_state(message_history=[m for m in message_history[:-1] if is_valid_user_history_message(m)])
            else:
                raise ValueError(f"Unexpected message type: {type(last_message)}")
            message = last_message
            trajectory = message_history
        else:
            user_state = user.get_init_state()
            first_message = deepcopy(DEFAULT_FIRST_AGENT_MESSAGE)
            first_message.timestamp = get_now()
            agent_state = agent.get_init_state(message_history=[first_message])
            trajectory = [first_message]
            message = first_message
            from_role = Role.AGENT
            to_role = Role.USER
        environment.sync_tools()
        state["tau2"] = Tau2BenchState(
            task=task, agent=agent, agent_state=agent_state,
            user=user, user_state=user_state, environment=environment,
            trajectory=trajectory, message=message,
            from_role=from_role, to_role=to_role,
            done=done, termination_reason=termination_reason,
            step_count=0, num_errors=0,
        )
        state["num_assistant_tool_calls"] = 0
        state["num_user_tool_calls"] = 0

    def _add_timestamps(self, message_history):
        time_offset = datetime.now() - timedelta(seconds=len(message_history))
        for i, msg in enumerate(message_history):
            msg.timestamp = format_time(time_offset + timedelta(seconds=i))
        return message_history

    async def _step(self, messages, state, **kwargs):
        new_messages = []
        tau2 = cast(Tau2BenchState, state["tau2"])
        if tau2["from_role"] in [Role.AGENT, Role.ENV] and tau2["to_role"] == Role.USER:
            tau2_user_msg, tau2["user_state"] = await self._run_in_thread(
                tau2["user"].generate_next_message, tau2["message"], tau2["user_state"]
            )
            try:
                tau2_user_msg.validate()
            except ValueError:
                tau2["done"] = True
                tau2["termination_reason"] = TerminationReason.USER_ERROR
                tau2["trajectory"].append(tau2_user_msg)
                return new_messages
            if UserSimulator.is_stop(tau2_user_msg):
                tau2["done"] = True
                tau2["termination_reason"] = TerminationReason.USER_STOP
            user_msg = vf.UserMessage(
                content=tau2_user_msg.content or f"Called {', '.join([tc.name for tc in tau2_user_msg.tool_calls or []])}",
            )
            state["num_user_tool_calls"] = state.get("num_user_tool_calls", 0) + len(tau2_user_msg.tool_calls or [])
            if not tau2_user_msg.is_tool_call():
                new_messages.append(user_msg)
            tau2["trajectory"].append(tau2_user_msg)
            tau2["message"] = tau2_user_msg
            tau2["from_role"] = Role.USER
            tau2["to_role"] = Role.ENV if tau2_user_msg.is_tool_call() else Role.AGENT
        elif tau2["from_role"] in [Role.USER, Role.AGENT] and tau2["to_role"] == Role.ENV:
            tau2_tool_msgs = []
            for tau2_tc in getattr(tau2["message"], "tool_calls", []):
                assert isinstance(tau2_tc, ToolCall)
                tau2_tool_msg = tau2["environment"].get_response(tau2_tc)
                if tau2_tool_msg.error:
                    tau2["num_errors"] += 1
                tau2_tool_msgs.append(tau2_tool_msg)
                if tau2["from_role"] == Role.AGENT:
                    tool_msg = vf.ToolMessage(tool_call_id=tau2_tc.id, content=tau2_tool_msg.content or "")
                    new_messages.append(tool_msg)
            tau2["trajectory"].extend(tau2_tool_msgs)
            if len(tau2_tool_msgs) > 1:
                tau2["message"] = MultiToolMessage(role="tool", tool_messages=tau2_tool_msgs)
            else:
                tau2["message"] = tau2_tool_msgs[0]
            tau2["to_role"] = tau2["from_role"]
            tau2["from_role"] = Role.ENV
        else:
            raise ValueError(f"Invalid from_role={tau2['from_role']} to_role={tau2['to_role']}")
        tau2["step_count"] += 1
        tau2["environment"].sync_tools()
        return new_messages


# ============================================================
# Entry point
# ============================================================

def load_environment(
    domain: str = "telecom",
    user_model: str = DEFAULT_USER_MODEL,
    user_args: dict = DEFAULT_LLM_ARGS_USER,
    user_base_url: str = DEFAULT_USER_BASE_URL,
    user_api_key_var: str = DEFAULT_USER_API_KEY_VAR,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_errors: int = DEFAULT_MAX_ERRORS,
    max_workers: int = DEFAULT_MAX_WORKERS,
    task_ids: list[int] | None = None,
    **kwargs,
) -> vf.MultiTurnEnv:
    download_tau2_data()
    env = Tau2BenchSeparatedEnv(
        domain=domain,
        user_model=user_model,
        user_args=user_args,
        user_base_url=user_base_url,
        user_api_key_var=user_api_key_var,
        max_steps=max_steps,
        max_errors=max_errors,
        max_workers=max_workers,
        task_ids=task_ids,
        **kwargs,
    )
    env.rubric.add_metric(prefix_stability_metric, weight=0.0)
    env.rubric.add_metric(append_ratio_metric, weight=0.0)
    env.rubric.add_metric(context_tokens_metric, weight=0.0)
    env.rubric.add_metric(total_edits_metric, weight=0.0)
    env.rubric.add_metric(max_list_length_metric, weight=0.0)
    return env
