"""Quick script to run explore env on longmemeval_m with proper JSON args."""
import subprocess, sys

cmd = [
    sys.executable, "-m", "verifiers.scripts.eval",
    "longmemeval-explore",
    "--model", "openai/gpt-5.4",
    "--provider", "prime",
    "-r", "1",
    "-s",
    "-n", "2",
    "--verbose",
    "--disable-env-server",
    "--temperature", "0",
    "--state-columns", "_final_answer,_turns_used,_context_list",
    "--env-args", '{"prompt_version": "notes", "variant": "longmemeval_m_cleaned"}',
]

subprocess.run(cmd)
