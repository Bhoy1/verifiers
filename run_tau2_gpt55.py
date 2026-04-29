"""Run tau2-bench context mode with gpt-5.5 on first 15 airline tasks."""
import subprocess, sys

cmd = [
    sys.executable, "-m", "verifiers.scripts.eval",
    "tau2-bench-context",
    "--model", "openai/gpt-5.5",
    "--provider", "prime",
    "-r", "1",
    "-s",
    "-n", "15",
    "--verbose",
    "--disable-env-server",
    "--temperature", "0",
    "--env-args", '{"domain": "airline"}',
]

subprocess.run(cmd)
