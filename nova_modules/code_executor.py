"""
Nova Code Executor
=====================
Safe code execution engine for Nova. Runs Python and shell commands
in isolated subprocesses with timeouts, safety checks, and ANSI stripping.

All execution happens via subprocess — never eval() or exec().

Usage:
    executor = NovaCodeExecutor()
    result = executor.execute_python("print('Hello from Nova!')")
    print(result["output"])
"""

import subprocess
import tempfile
import re
import time
import sys
import os
from pathlib import Path
from typing import Dict


# Regex to strip ANSI escape codes from subprocess output
ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')


class NovaCodeExecutor:
    """
    Safe code execution engine.

    - Python code runs in a subprocess (never eval/exec)
    - Shell commands are checked against a safety blocklist
    - All output is ANSI-stripped for clean display
    - 30-second timeout prevents runaway processes
    """

    # Commands that should never be run from the assistant
    BASH_BLOCKLIST = ["rm -rf", "format", "mkfs", "dd if="]

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    @staticmethod
    def _strip_ansi(text: str) -> str:
        """Remove ANSI escape codes from a string."""
        return ANSI_ESCAPE.sub('', text)

    # ------------------------------------------------------------------ #
    #  Python execution
    # ------------------------------------------------------------------ #

    def execute_python(self, code: str) -> Dict:
        """
        Run Python code in a subprocess with a timeout.

        Args:
            code: Python source code to execute.

        Returns:
            dict with keys: output, error, success, runtime_ms
        """
        start = time.perf_counter()
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=tempfile.gettempdir(),
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            stdout = self._strip_ansi(result.stdout)
            stderr = self._strip_ansi(result.stderr)

            return {
                "output": stdout,
                "error": stderr,
                "success": result.returncode == 0,
                "runtime_ms": round(elapsed_ms, 2),
            }

        except subprocess.TimeoutExpired:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return {
                "output": "",
                "error": f"Execution timed out after {self.timeout}s",
                "success": False,
                "runtime_ms": round(elapsed_ms, 2),
            }
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return {
                "output": "",
                "error": f"Execution failed: {e}",
                "success": False,
                "runtime_ms": round(elapsed_ms, 2),
            }

    # ------------------------------------------------------------------ #
    #  Shell / Bash execution
    # ------------------------------------------------------------------ #

    def execute_bash(self, command: str) -> Dict:
        """
        Run a shell command in a subprocess with safety checks.

        Args:
            command: Shell command string to execute.

        Returns:
            dict with keys: output, error, success
        """
        # Safety check against blocklist
        command_lower = command.lower()
        for blocked in self.BASH_BLOCKLIST:
            if blocked in command_lower:
                return {
                    "output": "",
                    "error": f"Blocked: command contains '{blocked}' which is not allowed",
                    "success": False,
                }

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                shell=True,
                cwd=tempfile.gettempdir(),
            )

            stdout = self._strip_ansi(result.stdout)
            stderr = self._strip_ansi(result.stderr)

            return {
                "output": stdout,
                "error": stderr,
                "success": result.returncode == 0,
            }

        except subprocess.TimeoutExpired:
            return {
                "output": "",
                "error": f"Command timed out after {self.timeout}s",
                "success": False,
            }
        except Exception as e:
            return {
                "output": "",
                "error": f"Command failed: {e}",
                "success": False,
            }

    # ------------------------------------------------------------------ #
    #  Write-and-run (temp file approach)
    # ------------------------------------------------------------------ #

    def write_and_run(self, code: str, filename: str = "nova_temp.py") -> Dict:
        """
        Write code to a temporary file, execute it, then clean up.

        Args:
            code: Python source code to execute.
            filename: Name for the temp file (default: nova_temp.py).

        Returns:
            Same dict as execute_python.
        """
        temp_path = Path(tempfile.gettempdir()) / filename
        try:
            temp_path.write_text(code, encoding="utf-8")

            start = time.perf_counter()
            result = subprocess.run(
                [sys.executable, str(temp_path)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=tempfile.gettempdir(),
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            stdout = self._strip_ansi(result.stdout)
            stderr = self._strip_ansi(result.stderr)

            return {
                "output": stdout,
                "error": stderr,
                "success": result.returncode == 0,
                "runtime_ms": round(elapsed_ms, 2),
            }

        except subprocess.TimeoutExpired:
            return {
                "output": "",
                "error": f"Execution timed out after {self.timeout}s",
                "success": False,
                "runtime_ms": self.timeout * 1000,
            }
        except Exception as e:
            return {
                "output": "",
                "error": f"Write-and-run failed: {e}",
                "success": False,
                "runtime_ms": 0.0,
            }
        finally:
            # Always clean up the temp file
            if temp_path.exists():
                temp_path.unlink()

    # ------------------------------------------------------------------ #
    #  Package installation
    # ------------------------------------------------------------------ #

    def install_package(self, package: str) -> Dict:
        """
        Install a Python package via pip.

        Args:
            package: Package name (e.g. "requests", "numpy==1.24").

        Returns:
            dict with keys: output, error, success
        """
        # Basic safety: reject anything that looks like a shell injection
        if any(c in package for c in [";", "&", "|", "`", "$", "(", ")"]):
            return {
                "output": "",
                "error": f"Rejected: package name '{package}' contains unsafe characters",
                "success": False,
            }

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package],
                capture_output=True,
                text=True,
                timeout=120,  # pip can be slow
            )

            stdout = self._strip_ansi(result.stdout)
            stderr = self._strip_ansi(result.stderr)

            return {
                "output": stdout,
                "error": stderr,
                "success": result.returncode == 0,
            }

        except subprocess.TimeoutExpired:
            return {
                "output": "",
                "error": "pip install timed out after 120s",
                "success": False,
            }
        except Exception as e:
            return {
                "output": "",
                "error": f"pip install failed: {e}",
                "success": False,
            }
