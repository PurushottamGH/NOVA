"""
Nova File System
===================
File and directory operations for Nova.
Provides safe read, write, edit, list, search, and script execution.

No external dependencies — uses only os, pathlib, subprocess.

Usage:
    fs = NovaFileSystem()
    content = fs.read_file("model/config.py")
    results = fs.search_in_files("training/", "learning_rate")
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional


class NovaFileSystem:
    """
    File system operations for Nova.

    - Read/write/edit text files
    - List directories with sizes
    - Search across files by content
    - Execute Python and shell scripts safely
    """

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    # ------------------------------------------------------------------ #
    #  Read
    # ------------------------------------------------------------------ #

    def read_file(self, path: str) -> Dict:
        """
        Read a text file and return its content with metadata.

        Args:
            path: Path to the file to read.

        Returns:
            dict with keys: content, lines, size_bytes, error
        """
        try:
            p = Path(path).resolve()
            if not p.exists():
                return {
                    "content": "",
                    "lines": 0,
                    "size_bytes": 0,
                    "error": f"File not found: {p}",
                }
            if not p.is_file():
                return {
                    "content": "",
                    "lines": 0,
                    "size_bytes": 0,
                    "error": f"Not a file: {p}",
                }

            size = p.stat().st_size
            content = p.read_text(encoding="utf-8", errors="replace")
            line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)

            return {
                "content": content,
                "lines": line_count,
                "size_bytes": size,
                "error": None,
            }
        except Exception as e:
            return {
                "content": "",
                "lines": 0,
                "size_bytes": 0,
                "error": str(e),
            }

    # ------------------------------------------------------------------ #
    #  Write
    # ------------------------------------------------------------------ #

    def write_file(self, path: str, content: str) -> Dict:
        """
        Write content to a file, creating parent directories if needed.

        Args:
            path: Destination file path.
            content: Text content to write.

        Returns:
            dict with keys: success, path, error
        """
        try:
            p = Path(path).resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return {
                "success": True,
                "path": str(p),
                "error": None,
            }
        except Exception as e:
            return {
                "success": False,
                "path": str(Path(path).resolve()),
                "error": str(e),
            }

    # ------------------------------------------------------------------ #
    #  Edit (search and replace)
    # ------------------------------------------------------------------ #

    def edit_file(self, path: str, old_text: str, new_text: str) -> Dict:
        """
        Find old_text in a file and replace it with new_text.

        Args:
            path: Path to the file to edit.
            old_text: Exact text to find.
            new_text: Replacement text.

        Returns:
            dict with keys: success, replacements, error
        """
        try:
            p = Path(path).resolve()
            if not p.exists():
                return {
                    "success": False,
                    "replacements": 0,
                    "error": f"File not found: {p}",
                }

            content = p.read_text(encoding="utf-8", errors="replace")
            count = content.count(old_text)

            if count == 0:
                return {
                    "success": False,
                    "replacements": 0,
                    "error": "old_text not found in file",
                }

            new_content = content.replace(old_text, new_text)
            p.write_text(new_content, encoding="utf-8")

            return {
                "success": True,
                "replacements": count,
                "error": None,
            }
        except Exception as e:
            return {
                "success": False,
                "replacements": 0,
                "error": str(e),
            }

    # ------------------------------------------------------------------ #
    #  List directory
    # ------------------------------------------------------------------ #

    def list_directory(self, path: str) -> Dict:
        """
        List files and folders in a directory with sizes.

        Args:
            path: Directory path to list.

        Returns:
            dict with keys: files, folders, total_files
        """
        try:
            p = Path(path).resolve()
            if not p.exists():
                return {
                    "files": [],
                    "folders": [],
                    "total_files": 0,
                    "error": f"Directory not found: {p}",
                }
            if not p.is_dir():
                return {
                    "files": [],
                    "folders": [],
                    "total_files": 0,
                    "error": f"Not a directory: {p}",
                }

            files = []
            folders = []

            for entry in sorted(p.iterdir()):
                if entry.is_file():
                    size = entry.stat().st_size
                    files.append({
                        "name": entry.name,
                        "size_bytes": size,
                        "size_human": self._human_size(size),
                    })
                elif entry.is_dir():
                    # Count children for context
                    try:
                        child_count = sum(1 for _ in entry.iterdir())
                    except PermissionError:
                        child_count = -1
                    folders.append({
                        "name": entry.name,
                        "children": child_count,
                    })

            return {
                "files": files,
                "folders": folders,
                "total_files": len(files),
            }
        except Exception as e:
            return {
                "files": [],
                "folders": [],
                "total_files": 0,
                "error": str(e),
            }

    # ------------------------------------------------------------------ #
    #  Run script
    # ------------------------------------------------------------------ #

    def run_script(self, path: str) -> Dict:
        """
        Execute a .py or .sh script file.

        Args:
            path: Path to the script to run.

        Returns:
            dict with keys: output, error, success, runtime_ms
        """
        try:
            p = Path(path).resolve()
            if not p.exists():
                return {
                    "output": "",
                    "error": f"Script not found: {p}",
                    "success": False,
                    "runtime_ms": 0.0,
                }

            suffix = p.suffix.lower()
            if suffix == ".py":
                cmd = [sys.executable, str(p)]
            elif suffix in (".sh", ".bash"):
                cmd = ["bash", str(p)]
            else:
                return {
                    "output": "",
                    "error": f"Unsupported script type: {suffix}",
                    "success": False,
                    "runtime_ms": 0.0,
                }

            start = time.perf_counter()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(p.parent),
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            return {
                "output": result.stdout,
                "error": result.stderr,
                "success": result.returncode == 0,
                "runtime_ms": round(elapsed_ms, 2),
            }

        except subprocess.TimeoutExpired:
            return {
                "output": "",
                "error": f"Script timed out after {self.timeout}s",
                "success": False,
                "runtime_ms": self.timeout * 1000,
            }
        except Exception as e:
            return {
                "output": "",
                "error": f"Script execution failed: {e}",
                "success": False,
                "runtime_ms": 0.0,
            }

    # ------------------------------------------------------------------ #
    #  Search in files
    # ------------------------------------------------------------------ #

    def search_in_files(
        self, directory: str, query: str, extension: str = ".py"
    ) -> List[Dict]:
        """
        Search for a query string across all files with a given extension.

        Args:
            directory: Root directory to search.
            query: Text to search for (case-sensitive).
            extension: File extension filter (e.g. ".py", ".txt").

        Returns:
            List of dicts with keys: file, line_number, line_content
        """
        results = []
        try:
            root = Path(directory).resolve()
            if not root.is_dir():
                return results

            pattern = f"**/*{extension}"
            for filepath in sorted(root.glob(pattern)):
                if not filepath.is_file():
                    continue
                # Skip hidden dirs and __pycache__
                parts = filepath.relative_to(root).parts
                if any(part.startswith(".") or part == "__pycache__" for part in parts):
                    continue

                try:
                    lines = filepath.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()
                except Exception:
                    continue

                for line_num, line in enumerate(lines, start=1):
                    if query in line:
                        results.append({
                            "file": str(filepath),
                            "line_number": line_num,
                            "line_content": line.strip(),
                        })

        except Exception:
            pass

        return results

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _human_size(size_bytes: int) -> str:
        """Convert byte count to human-readable string."""
        for unit in ("B", "KB", "MB", "GB"):
            if abs(size_bytes) < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"
