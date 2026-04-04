"""
Nova Tool Router
===================
Central router that detects user intent and dispatches to the right tool.
Acts as the orchestration layer between the chat engine and all Nova modules.

Supported intents:
    code    -> NovaCodeExecutor
    math    -> NovaMathEngine
    search  -> NovaWebSearch
    file    -> NovaFileSystem
    blender -> NovaBlenderAgent
    chat    -> passthrough to NovaMind

Usage:
    router = NovaToolRouter(model, tokenizer, config)
    result = router.route("solve x^2 - 4 = 0")
    # -> {"intent": "math", "result": "Solutions: [-2, 2]", "tool_output": {...}}
"""

import re
from typing import Dict, Optional


from collections import Counter


class NovaToolRouter:
    """
    Detects user intent from natural language and routes to the
    appropriate Nova tool module.
    """

    # Keyword -> intent mapping (checked in priority order)
    INTENT_KEYWORDS = {
        "code": [
            "write code", "python", "script", "function",
            "debug", "error", "fix", "run this", "execute",
            "def ", "class ", "import ", "how to build", "how to make",
        ],
        "math": [
            "solve", "calculate", "integrate", "derivative",
            "equation", "differentiate", "factor", "simplify",
        ],
        "search": [
            "search", "find online", "latest", "what is",
            "research", "look up", "google", "who is",
        ],
        "file": [
            "read file", "edit file", "open file", "save file",
            "directory", "folder", "list files", "write file",
        ],
        "blender": [
            "blender", "3d", "render", "mesh", "material",
            "animation", "bpy", "3d model",
        ],
    }

    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        # Lazy-import and initialise each tool
        from nova_modules.code_executor import NovaCodeExecutor
        from nova_modules.math_engine import NovaMathEngine
        from nova_modules.file_system import NovaFileSystem
        from nova_modules.blender_agent import NovaBlenderAgent

        self.code_executor = NovaCodeExecutor()
        self.math_engine = NovaMathEngine()
        self.file_system = NovaFileSystem()
        self.blender_agent = NovaBlenderAgent()

        # Web search is optional (needs requests + bs4)
        try:
            from nova_modules.web_search import NovaWebSearch
            self.web_search = NovaWebSearch()
        except ImportError:
            self.web_search = None
            print("[ToolRouter] NovaWebSearch unavailable (install requests + beautifulsoup4)")

        print("[ToolRouter] All tools initialised")

    # ------------------------------------------------------------------ #
    #  Intent detection
    # ------------------------------------------------------------------ #

    def detect_intent(self, user_message: str) -> str:
        """
        Classify user message into an intent category.

        Args:
            user_message: Raw user input string.

        Returns:
            One of: "code", "math", "search", "file", "blender", "chat"
        """
        text_lower = user_message.lower()

        for intent, keywords in self.INTENT_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    return intent

        # Check for inline code patterns (backticks or equals sign with variables)
        if re.search(r'```', user_message):
            return "code"
        if re.search(r'[a-zA-Z]\s*\^?\d*\s*[+\-*/=]', user_message):
            return "math"

        return "chat"

    # ------------------------------------------------------------------ #
    #  Garbage Detection & Fallbacks
    # ------------------------------------------------------------------ #

    def is_garbage(self, text: str) -> bool:
        """
        Detect model degradation (hallucinations, token merging bugs).
        """
        if not text or len(text.strip()) < 5:
            return True
        words = text.split()
        if not words:
            return True
        
        # 1. Merged words detection (tokenizer bug)
        avg_word_len = sum(len(w) for w in words) / len(words)
        if avg_word_len > 12:
            return True

        # 2. Word repetition detection (hallucination loop)
        word_counts = Counter(words)
        if word_counts.most_common(1)[0][1] > 5:
            return True

        # 3. Dense text without whitespace
        if len(text) > 80 and text.count(' ') < 8:
            return True

        return False

    def _code_template_response(self, message: str) -> str:
        """Provide a hardcoded high-quality code template for common tasks."""
        msg_lower = message.lower()
        if "calculator" in msg_lower:
            return '''Here is a Python calculator:

```python
def calculator():
    print("Nova Calculator")
    print("Operations: +, -, *, /")
    
    while True:
        try:
            a = float(input("First number: "))
            op = input("Operation (+,-,*,/): ")
            b = float(input("Second number: "))
            
            if op == "+": print(f"Result: {a + b}")
            elif op == "-": print(f"Result: {a - b}")
            elif op == "*": print(f"Result: {a * b}")
            elif op == "/": 
                if b != 0: print(f"Result: {a / b}")
                else: print("Error: Division by zero")
            else:
                print("Unknown operation")
        except ValueError:
            print("Invalid input")
        
        if input("Continue? (y/n): ").lower() != "y":
            break

calculator()
```'''
        return (
            "I can help with that. Could you be more specific?\n"
            "Example: 'write a Python function that sorts a list'"
        )

    # ------------------------------------------------------------------ #
    #  Main router
    # ------------------------------------------------------------------ #

    def route(self, user_message: str, response: Optional[str] = None) -> Dict:
        """
        Detect intent and dispatch to the appropriate tool.
        If a response is provided, it validates it for 'garbage' loops.
        """
        intent = self.detect_intent(user_message)
        text_lower = user_message.lower()

        # If we are post-processing a model response
        if response:
            if self.is_garbage(response):
                if intent == "code":
                    response = self._code_template_response(user_message)
                elif intent == "search" and self.web_search:
                    response = self.web_search.search_and_summarize(user_message)
                else:
                    response = (
                        "My training is still in progress for this query.\n"
                        "Try: /search " + user_message + " for web results."
                    )
            return {
                "intent": intent,
                "result": response,
                "tool_output": {"validated": True}
            }

        # Otherwise, standard tool routing
        if intent == "code":
            # Check for "how to build/make" templates
            build_match = re.search(
                r'how (?:to|can i) (?:build|make|create|write|code) (?:a |an )?(.+)',
                text_lower
            )
            if build_match:
                topic = build_match.group(1).strip()
                result = self._code_template_response(user_message)
                tool_output = {"topic": topic, "template": True}
            else:
                tool_output = self.execute_code_request(user_message)
                result = self._format_code_result(tool_output)

        elif intent == "math":
            answer = self.math_engine.detect_and_solve(user_message)
            if answer:
                tool_output = {"answer": answer}
                result = answer
            else:
                # Fallback to chat if math engine fails
                intent = "chat"
                tool_output = {}
                result = None

        elif intent == "search":
            tool_output = self.execute_search_request(user_message)
            context = tool_output.get("context", "")
            if context and "No results found" not in context:
                # Instruct model to summarize the provided context
                result = f"Summarize this: {context}"
            else:
                result = "No online search results found."

        elif intent == "file":
            tool_output = self.execute_file_request(user_message)
            result = self._format_file_result(tool_output)

        elif intent == "blender":
            tool_output = self.execute_blender_request(user_message)
            result = tool_output.get("script", "Could not generate Blender script.")

        else:
            # chat — no tool needed, pass through to model
            tool_output = {}
            result = None

        return {
            "intent": intent,
            "result": result,
            "tool_output": tool_output,
        }

    # ------------------------------------------------------------------ #
    #  Code execution
    # ------------------------------------------------------------------ #

    def execute_code_request(self, message: str) -> Dict:
        """
        Extract or generate code from the message and run it.

        Args:
            message: User message containing or describing code.

        Returns:
            Execution result dict from NovaCodeExecutor.
        """
        code = self._extract_code_from_message(message)

        if code:
            # User provided code directly — run it
            return self.code_executor.execute_python(code)
        else:
            # No code found — ask model to generate it
            return {
                "output": "",
                "error": "No executable code found in message. Try wrapping code in ```python ... ``` blocks.",
                "success": False,
                "runtime_ms": 0.0,
            }

    def _extract_code_from_message(self, message: str) -> Optional[str]:
        """Extract code block from user message."""
        # Try ```python ... ```
        match = re.search(r'```python\s*(.*?)```', message, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try ``` ... ```
        match = re.search(r'```\s*(.*?)```', message, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Single-line: "run: print('hello')" or "execute: 2+2"
        match = re.search(r'(?:run|execute)\s*:\s*(.+)', message, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        return None

    def _format_code_result(self, result: Dict) -> str:
        """Format code execution result for display."""
        if result.get("success"):
            output = result.get("output", "").strip()
            ms = result.get("runtime_ms", 0)
            return f"Output:\n{output}\n(completed in {ms:.0f}ms)"
        else:
            error = result.get("error", "Unknown error")
            return f"Error:\n{error}"

    # ------------------------------------------------------------------ #
    #  Web search
    # ------------------------------------------------------------------ #

    def execute_search_request(self, message: str) -> Dict:
        """
        Search the web and return summarised context.

        Args:
            message: User search query.

        Returns:
            dict with key "context" containing formatted results.
        """
        if not self.web_search:
            return {
                "context": "Web search is unavailable. Install requests and beautifulsoup4.",
            }

        # Strip common prefix words to get a cleaner query
        query = re.sub(
            r'^(search for|search|look up|find|what is|who is|research)\s+',
            '', message, flags=re.IGNORECASE,
        ).strip()

        if not query:
            query = message

        context = self.web_search.search_and_summarize(query)
        return {"context": context, "query": query}

    # ------------------------------------------------------------------ #
    #  File operations
    # ------------------------------------------------------------------ #

    def execute_file_request(self, message: str) -> Dict:
        """
        Parse and execute a file operation from the user message.

        Args:
            message: User message describing a file operation.

        Returns:
            Result dict from NovaFileSystem.
        """
        text_lower = message.lower()

        # Try to extract a file path from the message
        path_match = re.search(
            r'["\']([^"\']+)["\']|(\S+\.\w{1,5})', message
        )
        path = path_match.group(1) or path_match.group(2) if path_match else None

        if "read" in text_lower and path:
            return self.file_system.read_file(path)
        elif "list" in text_lower or "directory" in text_lower or "folder" in text_lower:
            target = path or "."
            return self.file_system.list_directory(target)
        elif "edit" in text_lower and path:
            return {"action": "edit", "path": path, "note": "Specify old_text and new_text for editing."}
        elif "write" in text_lower or "save" in text_lower:
            return {"action": "write", "path": path, "note": "Provide content to write."}
        else:
            return {"error": "Could not determine file operation. Try: read file 'path', list directory, etc."}

    def _format_file_result(self, result: Dict) -> str:
        """Format file operation result for display."""
        if "content" in result:
            lines = result.get("lines", 0)
            size = result.get("size_bytes", 0)
            preview = result["content"][:500]
            return f"File ({lines} lines, {size} bytes):\n{preview}"
        elif "files" in result:
            files = result.get("files", [])
            folders = result.get("folders", [])
            parts = []
            for f in folders:
                parts.append(f"  [DIR]  {f['name']}/")
            for f in files:
                parts.append(f"  [FILE] {f['name']} ({f.get('size_human', '')})")
            return "\n".join(parts) if parts else "Empty directory."
        elif "error" in result:
            return result["error"]
        elif "note" in result:
            return result["note"]
        else:
            return str(result)

    # ------------------------------------------------------------------ #
    #  Blender
    # ------------------------------------------------------------------ #

    def execute_blender_request(self, message: str) -> Dict:
        """
        Parse and execute a Blender request.

        Args:
            message: User message describing a 3D task.

        Returns:
            dict with generated script and/or execution result.
        """
        text_lower = message.lower()

        # Check for predefined object types
        for obj_type in ["cube", "sphere", "plane", "camera", "light"]:
            if obj_type in text_lower:
                script = self.blender_agent.create_object(obj_type)
                return {"script": script, "type": "template", "object": obj_type}

        # Check for effects
        for effect in ["glow", "motion_blur", "depth_of_field", "hdri_lighting"]:
            if effect.replace("_", " ") in text_lower or effect in text_lower:
                script = self.blender_agent.add_effect(effect)
                return {"script": script, "type": "effect", "effect": effect}

        # Fall back to AI generation
        try:
            script = self.blender_agent.generate_script(
                message, self.tokenizer, self.model
            )
            return {"script": script, "type": "generated"}
        except Exception as e:
            return {"script": "", "type": "error", "error": str(e)}
