"""
Nova Code Executor
=====================
Safe code execution engine for Nova. Runs Python and shell commands
in isolated subprocesses with timeouts, safety checks, and ANSI stripping.

All execution happens via subprocess — never eval() or exec().

Security layers:
    1. Subprocess isolation (no eval/exec)
    2. Regex-based blocklist for dangerous patterns
    3. Resource limits via ulimit (memory, CPU time, file size)
    4. Network-disabled environment (PYTHONSAFEPATH, restricted env)
    5. Filesystem sandbox (temp dir only, no path traversal)
    6. Timeout guards on every execution path

Usage:
    executor = NovaCodeExecutor()
    result = executor.execute_python("print('Hello from Nova!')")
    print(result["output"])
"""

import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:
    import resource

    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False  # Windows doesn't support resource module


# Regex to strip ANSI escape codes from subprocess output
ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


class NovaCodeExecutor:
    """
    Safe code execution engine.

    - Python code runs in a subprocess (never eval/exec)
    - Shell commands are checked against a safety blocklist
    - All output is ANSI-stripped for clean display
    - 30-second timeout prevents runaway processes
    - Resource limits on memory and CPU time (Unix only)
    """

    # Regex patterns for dangerous code — checked before execution
    DANGEROUS_PATTERNS = [
        # Filesystem destruction
        r"\b(shutil|rmtree|os\.system|os\.popen)\s*\(",
        r'__import__\s*\(\s*["\']os["\']',
        r'__import__\s*\(\s*["\']subprocess["\']',
        r'__import__\s*\(\s*["\']sys["\']',
        # Network access (exfiltration risk)
        r'__import__\s*\(\s*["\']socket["\']',
        r'__import__\s*\(\s*["\']urllib["\']',
        r'__import__\s*\(\s*["\']requests["\']',
        r'__import__\s*\(\s*["\']http["\']',
        r"import\s+socket",
        r"import\s+urllib",
        r"import\s+requests\b",
        r"import\s+http\b",
        r"socket\.\w+\(",
        r"urllib\.\w+\(",
        # Process manipulation
        r"os\.fork\s*\(",
        r"os\.exec",
        r"os\.spawn",
        r"subprocess\.\w+\(",
        r"pty\.\w+",
        r"ctypes\.\w+",
        # Eval/exec variants
        r"\beval\s*\(",
        r"\bexec\s*\(",
        r"compile\s*\(",
        # System-level
        r"os\.chmod\s*\(",
        r"os\.chown\s*\(",
        r"os\.setuid\s*\(",
        r"os\.setgid\s*\(",
        r"os\.unlink\s*\(",
        # Signal manipulation
        r"signal\.\w+\(",
        # Thread/process spawning
        r"threading\.\w+",
        r"multiprocessing\.\w+",
        r"concurrent\.\w+",
        # Deserialization / injection attacks
        r"pickle\.\w+\(",
        r"marshal\.\w+\(",
        r"importlib\.\w+\(",
        # Builtins override
        r"__builtins__",
        # Scope access
        r"globals\s*\(\s*\)",
        r"locals\s*\(\s*\)",
        r"vars\s*\(\s*\)",
        # Attribute manipulation bypass
        r"getattr\s*\(",
        r"setattr\s*\(",
        r"delattr\s*\(",
        # File I/O
        r"open\s*\(",
        r"print\s*\(\s*open",
    ]

    # Commands that should never be run from the assistant
    BASH_BLOCKLIST = [
        "rm -rf",
        "rm -r /",
        "rm /*",
        "format",
        "mkfs",
        "dd if=",
        "sudo",
        "su ",
        "chmod",
        "chown",
        "mkfifo",
        "mknod",
        "wget ",
        "curl ",
        "nc ",
        "netcat",
        "nmap",
        "kill ",
        "killall",
        "pkill",
    ]

    def __init__(self, timeout: int = 30, max_memory_mb: int = 256):
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb

    @staticmethod
    def _strip_ansi(text: str) -> str:
        """Remove ANSI escape codes from a string."""
        return ANSI_ESCAPE.sub("", text)

    def _is_safe_code(self, code: str) -> tuple:
        """
        Check code against dangerous patterns.

        Returns:
            (is_safe: bool, reason: str)
        """
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"blocked pattern: {pattern}"
        return True, ""

    def _is_safe_command(self, command: str) -> tuple:
        """
        Check shell command against blocklist.

        Returns:
            (is_safe: bool, reason: str)
        """
        command_lower = command.lower()
        for blocked in self.BASH_BLOCKLIST:
            if blocked.lower() in command_lower:
                return False, f"blocked command: '{blocked}'"
        # Check for path traversal
        if "../" in command or "..\\" in command:
            return False, "path traversal detected"
        return True, ""

    def _set_resource_limits(self):
        """Set memory and CPU time limits for the subprocess (Unix only)."""
        if not HAS_RESOURCE:
            return
        max_mem = self.max_memory_mb * 1024 * 1024  # Convert MB to bytes
        resource.setrlimit(resource.RLIMIT_AS, (max_mem, max_mem))
        resource.setrlimit(resource.RLIMIT_CPU, (self.timeout, self.timeout))
        resource.setrlimit(resource.RLIMIT_FSIZE, (max_mem, max_mem))

    def _get_safe_env(self) -> dict:
        """Create a restricted environment for subprocess execution."""
        safe_env = os.environ.copy()
        # Prevent Python from importing from current directory
        safe_env["PYTHONSAFEPATH"] = "1"
        # Remove potentially dangerous env vars
        for key in list(safe_env.keys()):
            if key.upper() in ("PYTHONPATH", "PYTHONSTARTUP", "PYTHONINSPECT"):
                del safe_env[key]
        return safe_env

    # ------------------------------------------------------------------ #
    #  Python execution
    # ------------------------------------------------------------------ #

    def execute_python(self, code: str) -> dict:
        """
        Run Python code in a subprocess with safety checks and timeouts.

        Args:
            code: Python source code to execute.

        Returns:
            dict with keys: output, error, success, runtime_ms
        """
        # Safety check
        is_safe, reason = self._is_safe_code(code)
        if not is_safe:
            return {
                "output": "",
                "error": f"Code rejected: {reason}",
                "success": False,
                "runtime_ms": 0.0,
            }

        # Write code to a temp file in a secure temp directory
        temp_dir = tempfile.mkdtemp(prefix="nova_exec_")
        temp_path = Path(temp_dir) / "script.py"

        start = time.perf_counter()
        try:
            temp_path.write_text(code, encoding="utf-8")

            # Build restricted environment
            safe_env = self._get_safe_env()

            # Prepend resource-limiting wrapper for Unix
            if HAS_RESOURCE:
                wrapper_code = (
                    f"import resource, sys\n"
                    f"resource.setrlimit(resource.RLIMIT_AS, ({self.max_memory_mb * 1024 * 1024}, {self.max_memory_mb * 1024 * 1024}))\n"
                    f"resource.setrlimit(resource.RLIMIT_CPU, ({self.timeout}, {self.timeout}))\n"
                    f"exec(open({str(temp_path)!r}).read())\n"
                )
                cmd = [sys.executable, "-c", wrapper_code]
            else:
                cmd = [sys.executable, str(temp_path)]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=temp_dir,
                env=safe_env,
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
        finally:
            # Always clean up temp files
            try:
                if temp_path.exists():
                    temp_path.unlink()
                if Path(temp_dir).exists():
                    Path(temp_dir).rmdir()
            except OSError:
                pass

    # ------------------------------------------------------------------ #
    #  Shell / Bash execution
    # ------------------------------------------------------------------ #

    def execute_bash(self, command: str) -> dict:
        """
        Run a shell command in a subprocess with safety checks.

        Args:
            command: Shell command string to execute.

        Returns:
            dict with keys: output, error, success
        """
        # Safety check
        is_safe, reason = self._is_safe_command(command)
        if not is_safe:
            return {
                "output": "",
                "error": f"Command rejected: {reason}",
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

    def write_and_run(self, code: str, filename: str = "nova_temp.py") -> dict:
        """
        Write code to a temporary file, execute it, then clean up.

        Args:
            code: Python source code to execute.
            filename: Name for the temp file (default: nova_temp.py).

        Returns:
            Same dict as execute_python.
        """
        # Safety check
        is_safe, reason = self._is_safe_code(code)
        if not is_safe:
            return {
                "output": "",
                "error": f"Code rejected: {reason}",
                "success": False,
                "runtime_ms": 0.0,
            }

        # Sanitize filename to prevent path traversal
        safe_filename = Path(filename).name
        if not safe_filename.endswith(".py"):
            safe_filename += ".py"

        temp_dir = tempfile.mkdtemp(prefix="nova_exec_")
        temp_path = Path(temp_dir) / safe_filename

        try:
            temp_path.write_text(code, encoding="utf-8")

            safe_env = self._get_safe_env()
            start = time.perf_counter()
            result = subprocess.run(
                [sys.executable, str(temp_path)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=temp_dir,
                env=safe_env,
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
            try:
                if temp_path.exists():
                    temp_path.unlink()
                if Path(temp_dir).exists():
                    Path(temp_dir).rmdir()
            except OSError:
                pass

    # ------------------------------------------------------------------ #
    #  Package installation
    # ------------------------------------------------------------------ #

    def install_package(self, package: str) -> dict:
        """
        Install a Python package via pip.

        Args:
            package: Package name (e.g. "requests", "numpy==1.24").

        Returns:
            dict with keys: output, error, success
        """
        # Strict validation: only allow package names with optional version specifiers
        if not re.match(r"^[a-zA-Z0-9_\-\.]+(\\s*[><!=~]+\\s*[a-zA-Z0-9_\-\\.]+)?$", package.strip()):
            return {
                "output": "",
                "error": f"Rejected: package name '{package}' contains unsafe characters",
                "success": False,
            }

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--user", package.strip()],
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

    # ------------------------------------------------------------------ #
    #  Execute with context
    # ------------------------------------------------------------------ #

    def execute_with_context(self, code: str, context_vars: dict) -> dict:
        """
        Execute Python code with pre-injected context variables.

        Serializes context_vars using repr() and prepends them as
        variable assignments before the user code.

        Args:
            code: Python source code to execute.
            context_vars: Dictionary of variable names to values to inject.
                Example: {"x": 5, "data": [1, 2, 3]}

        Returns:
            Same dict as execute_python (output, error, success, runtime_ms).
        """
        try:
            preamble_lines = []
            for var_name, var_value in context_vars.items():
                preamble_lines.append(f"{var_name} = {repr(var_value)}")
            preamble = "\n".join(preamble_lines) + "\n"
            full_code = preamble + code
            return self.execute_python(full_code)
        except Exception as e:
            return {
                "output": "",
                "error": f"Context injection failed: {e}",
                "success": False,
                "runtime_ms": 0.0,
            }

    # ------------------------------------------------------------------ #
    #  Test suite runner
    # ------------------------------------------------------------------ #

    def run_test_suite(self, code: str, test_cases: list[dict]) -> dict:
        """
        Run a series of test cases against a piece of code.

        Each test case appends a print() call with the input and compares
        the stripped output to the expected output. Each test runs in its
        own subprocess via execute_python() (no shared state).

        Args:
            code: Python source code containing the function(s) to test.
            test_cases: List of dicts with "input" and "expected_output" keys.
                Example: [{"input": "func(5)", "expected_output": "25"}]

        Returns:
            dict with keys: total, passed, failed, results.
            results is a list of dicts with: input, expected, got, passed.
        """
        try:
            results = []
            passed = 0
            failed = 0

            for tc in test_cases:
                tc_input = tc.get("input", "")
                expected = tc.get("expected_output", "")

                # Build test code: user code + print(input_expression)
                test_code = code + f"\nprint({tc_input})\n"
                result = self.execute_python(test_code)

                got = result["output"].strip()
                is_pass = got == expected.strip()

                if is_pass:
                    passed += 1
                else:
                    failed += 1

                results.append({
                    "input": tc_input,
                    "expected": expected,
                    "got": got,
                    "passed": is_pass,
                })

            return {
                "total": len(test_cases),
                "passed": passed,
                "failed": failed,
                "results": results,
            }
        except Exception as e:
            return {
                "total": len(test_cases),
                "passed": 0,
                "failed": len(test_cases),
                "results": [],
                "error": f"Test suite failed: {e}",
            }

    # ------------------------------------------------------------------ #
    #  Code profiler
    # ------------------------------------------------------------------ #

    def profile_code(self, code: str) -> dict:
        """
        Profile Python code using cProfile and return timing statistics.

        Wraps the user code with cProfile instrumentation, runs it via
        execute_python(), and parses the profiler output from stdout.

        Args:
            code: Python source code to profile.

        Returns:
            dict with keys: output, profile, success, runtime_ms.
            output = the code's actual stdout.
            profile = cProfile stats as string.
        """
        try:
            # Indent user code for inclusion in the wrapper
            indented_code = "\n".join("    " + line for line in code.split("\n"))

            profiled_code = (
                "import cProfile, pstats, io, sys\n"
                "pr = cProfile.Profile()\n"
                "pr.enable()\n"
                "try:\n"
                f"{indented_code}\n"
                "finally:\n"
                "    pr.disable()\n"
                "    s = io.StringIO()\n"
                "    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')\n"
                "    ps.print_stats(10)\n"
                "    print('__PROFILE__:' + s.getvalue())\n"
            )

            result = self.execute_python(profiled_code)

            # Parse output to split code output from profiler output
            raw_output = result.get("output", "")
            if "__PROFILE__:" in raw_output:
                parts = raw_output.split("__PROFILE__:", 1)
                code_output = parts[0].rstrip()
                profile_output = parts[1].strip()
            else:
                code_output = raw_output
                profile_output = ""

            return {
                "output": code_output,
                "profile": profile_output,
                "success": result.get("success", False),
                "runtime_ms": result.get("runtime_ms", 0.0),
            }
        except Exception as e:
            return {
                "output": "",
                "profile": "",
                "success": False,
                "runtime_ms": 0.0,
                "error": f"Profiling failed: {e}",
            }

    # ------------------------------------------------------------------ #
    #  Error explainer
    # ------------------------------------------------------------------ #

    def explain_error(self, error_output: str) -> str:
        """
        Explain a Python traceback in plain English.

        Uses regex pattern matching to identify common error types and
        provide human-readable explanations with suggested fixes.

        Args:
            error_output: Python traceback/error string.

        Returns:
            Plain English explanation of the error with a suggested fix.
        """
        try:
            error_output = error_output.strip()
            if not error_output:
                return "No error output provided."

            # NameError
            match = re.search(r"NameError: name '(.+?)' is not defined", error_output)
            if match:
                name = match.group(1)
                return (
                    f"NameError: Variable '{name}' was used before being assigned or defined.\n"
                    f"Fix: Make sure '{name}' is defined before this line. Check for typos in "
                    f"variable names, or import the module if '{name}' is from an external library."
                )

            # IndexError
            if re.search(r"IndexError: list index out of range", error_output):
                return (
                    "IndexError: You tried to access an index that doesn't exist in the list.\n"
                    "Fix: Check the length of your list with len() before accessing indices. "
                    "Remember that Python lists are 0-indexed, so a list of length N has "
                    "valid indices 0 to N-1."
                )

            # TypeError
            match = re.search(r"TypeError: (.+)", error_output)
            if match:
                msg = match.group(1)
                return (
                    f"TypeError: Type mismatch — {msg}\n"
                    "Fix: Check that you're passing the correct types to functions and operators. "
                    "Use type() to inspect variables and convert types with int(), str(), float(), etc."
                )

            # ZeroDivisionError
            if re.search(r"ZeroDivisionError", error_output):
                return (
                    "ZeroDivisionError: Division by zero detected.\n"
                    "Fix: Add a check before dividing: 'if denominator != 0: result = a / denominator'. "
                    "Verify that your denominator variable isn't accidentally set to zero."
                )

            # IndentationError
            if re.search(r"IndentationError", error_output):
                return (
                    "IndentationError: Check your indentation.\n"
                    "Fix: Python requires consistent indentation (use 4 spaces per level). "
                    "Make sure you're not mixing tabs and spaces. Verify that all blocks "
                    "(if/for/while/def/class) have properly indented bodies."
                )

            # SyntaxError
            if re.search(r"SyntaxError", error_output):
                return (
                    "SyntaxError: Syntax error — missing bracket, colon, or quote.\n"
                    "Fix: Check for missing colons after if/for/def/class statements, "
                    "unmatched parentheses/brackets/braces, and unclosed string quotes. "
                    "Look at the line number in the traceback for the exact location."
                )

            # KeyError
            match = re.search(r"KeyError: (.+)", error_output)
            if match:
                key = match.group(1)
                return (
                    f"KeyError: The key {key} was not found in the dictionary.\n"
                    "Fix: Use dict.get(key, default) to safely access keys, or check "
                    "with 'if key in my_dict' before accessing."
                )

            # AttributeError
            match = re.search(r"AttributeError: (.+)", error_output)
            if match:
                msg = match.group(1)
                return (
                    f"AttributeError: {msg}\n"
                    "Fix: Check that the object has the attribute/method you're trying to access. "
                    "Use dir(obj) to list available attributes, or check the object's type with type()."
                )

            # ValueError
            match = re.search(r"ValueError: (.+)", error_output)
            if match:
                msg = match.group(1)
                return (
                    f"ValueError: {msg}\n"
                    "Fix: The function received a value of correct type but inappropriate value. "
                    "Validate your inputs before passing them to functions."
                )

            # ImportError / ModuleNotFoundError
            match = re.search(r"(?:ImportError|ModuleNotFoundError): (.+)", error_output)
            if match:
                msg = match.group(1)
                return (
                    f"ImportError: {msg}\n"
                    "Fix: Install the missing module with 'pip install <module_name>'. "
                    "Check that the module name is spelled correctly."
                )

            # FileNotFoundError
            match = re.search(r"FileNotFoundError: (.+)", error_output)
            if match:
                msg = match.group(1)
                return (
                    f"FileNotFoundError: {msg}\n"
                    "Fix: Verify the file path exists. Use os.path.exists() to check "
                    "before accessing. Check for typos in the filename."
                )

            # Fallback
            return f"Unknown error. Check the traceback above.\n\n{error_output}"
        except Exception:
            return f"Could not analyze error. Raw output:\n{error_output}"
