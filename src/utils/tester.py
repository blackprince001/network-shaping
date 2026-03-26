"""
src/utils/tester.py

Discovers and runs the __main__ blocks in each src module as unit tests.
Used by the `test` subcommand in main.py.
"""

import subprocess
import sys
from pathlib import Path

# Each entry is (display_name, module_path_relative_to_project_root)
_TEST_MODULES = [
    ("pipe_bridge",       "src/utils/pipe_bridge.py"),
    ("metrics",           "src/environments/metrics.py"),
    ("ns3_env (mock)",    "src/environments/ns3_env.py"),
    ("ppo_agent (mock)",  "src/agents/ppo_agent.py"),
]


def run_all_tests(project_root: Path) -> bool:
    """Run every test module. Returns True if all passed."""
    passed = 0
    failed = 0

    for name, rel_path in _TEST_MODULES:
        module_path = project_root / rel_path
        print(f"\n{'─' * 50}")
        print(f"  Running: {name}")
        print(f"{'─' * 50}")

        result = subprocess.run(
            [sys.executable, str(module_path)],
            cwd=str(project_root),
        )

        if result.returncode == 0:
            print(f"  ✓  {name} passed")
            passed += 1
        else:
            print(f"  ✗  {name} FAILED (exit code {result.returncode})")
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'=' * 50}")
    return failed == 0
