"""
Nova Terminal Interface
==========================
Premium terminal user interface for the NovaMind project.
Uses Rich for high-fidelity formatting and Prompt Toolkit for interaction.

Features:
- Capabilities Menu: Code, Math, Search, Files, Blender, Chat
- Intent Routing: Automated tool selection via NovaToolRouter
- Syntax Highlighting: Beautiful code block rendering
- Interactive Loop: History support and intuitive commands
- Performance Metrics: Real-time token and response-time tracking
"""

import sys
import time
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style as PromptStyle
from rich.box import ROUNDED
from rich.columns import Columns

# Third-party dependencies
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.text import Text
from rich.theme import Theme

# Nova project imports
from inference.chat import NovaChatEngine
from nova_modules.tool_router import NovaToolRouter

# --- UI Configuration ---
CUSTOM_THEME = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "bold green",
        "user": "bold blue",
        "nova": "bold magenta",
        "tool": "bold cyan",
    }
)

console = Console(theme=CUSTOM_THEME)

# --- Commands ---
COMMANDS = {
    "/run": "Execute Python code directly",
    "/search": "Force a web search for a query",
    "/file": "Read a specific file",
    "/blender": "Generate a Blender script from description",
    "/clear": "Clear conversation history",
    "/exit": "Close the session",
}


# --- Initialization ---
def display_banner():
    """Show the NovaMind startup banner."""
    banner_text = Text("\n" + "=" * 60 + "\n", style="nova")
    banner_text.append("   _  __               __  ___ _             __\n", style="bold magenta")
    banner_text.append("  / |/ /___  _  __ ___ _  /  |/ /(_)___  ___/ /\n", style="bold magenta")
    banner_text.append(" /    // _ \\| |/ // _ `/ / /|_/ // // _ \\/ _  / \n", style="bold magenta")
    banner_text.append(
        "/_/|_/ \\___/|___/ \\_,_/ /_/  /_//_//_//_/\\_,_/  \n", style="bold magenta"
    )
    banner_text.append("=" * 60 + "\n", style="nova")

    console.print(banner_text)
    console.print(
        Panel(
            "Welcome, Purushottam. I am [bold magenta]Nova[/], your intelligent assistant.",
            border_style="magenta",
            box=ROUNDED,
        )
    )


def display_menu():
    """Show the capabilities menu."""
    menu_items = [
        "[1] [bold cyan]Code[/]",
        "[2] [bold green]Math[/]",
        "[3] [bold yellow]Search[/]",
        "[4] [bold blue]Files[/]",
        "[5] [bold magenta]Blender[/]",
        "[6] [bold white]Chat[/]",
    ]
    console.print(Columns(menu_items, width=15, equal=True), justify="center")
    console.print("-" * 60, style="nova", justify="center")


def main():
    display_banner()

    # Initialization Spinner
    with console.status("[bold magenta]Igniting NovaMind...[/]", spinner="dots"):
        try:
            # Resolve paths relative to this script's location
            project_root = Path(__file__).parent.resolve()
            model_path = str(project_root / "weights" / "final_model")
            tokenizer_path = str(project_root / "weights" / "tokenizer")

            chat_engine = NovaChatEngine(model_path, tokenizer_path)
            # Re-use engine's model/tokenizer for the router
            router = NovaToolRouter(
                chat_engine.model, chat_engine.tokenizer, chat_engine.model.config
            )

            console.print("[success]NovaMind Online.[/]\n")
        except Exception as e:
            console.print(f"[error]Initialization failed: {e}[/]")
            sys.exit(1)

    display_menu()

    # Session setup
    session = PromptSession(
        history=FileHistory(".nova_history"),
        auto_suggest=AutoSuggestFromHistory(),
    )

    # Prompt style
    ps_style = PromptStyle.from_dict(
        {
            "prompt": "ansicyan bold",
        }
    )

    while True:
        try:
            user_input = session.prompt("Nova > ", style=ps_style).strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                cmd_parts = user_input.split(" ", 1)
                cmd = cmd_parts[0].lower()
                arg = cmd_parts[1] if len(cmd_parts) > 1 else ""

                if cmd == "/exit":
                    console.print("[info]Closing session. Goodbye, Purushottam.[/]")
                    break
                elif cmd == "/clear":
                    chat_engine.reset_history()
                    console.clear()
                    display_banner()
                    display_menu()
                    continue
                elif cmd == "/run":
                    if not arg:
                        console.print("[warning]Usage: /run <python code>[/]")
                        continue
                    # Force code execution intent
                    intent_result = router.execute_code_request(arg)
                    _display_tool_output("code", intent_result)
                    continue
                elif cmd == "/search":
                    if not arg:
                        console.print("[warning]Usage: /search <query>[/]")
                        continue
                    intent_result = router.execute_search_request(arg)
                    _display_tool_output("search", intent_result)
                    continue
                elif cmd == "/file":
                    if not arg:
                        console.print("[warning]Usage: /file <path>[/]")
                        continue
                    intent_result = router.execute_file_request(f"read file {arg}")
                    _display_tool_output("file", intent_result)
                    continue
                elif cmd == "/blender":
                    if not arg:
                        console.print("[warning]Usage: /blender <description>[/]")
                        continue
                    intent_result = router.execute_blender_request(arg)
                    _display_tool_output("blender", intent_result)
                    continue
                else:
                    console.print(f"[error]Unknown command: {cmd}[/]")
                    continue

            # Standard processing
            start_time = time.time()

            with Live(
                Spinner("bouncingBar", text="[magenta]Processing...[/]"), refresh_per_second=10
            ) as live:
                # 1. Routing
                route_data = router.route(user_input)
                intent = route_data["intent"]
                tool_msg = route_data["result"]

                # Update live display for intent
                if intent != "chat":
                    live.update(f"[cyan]Intent detected: [bold]{intent.upper()}[/][/]")

                # 2. Final Chat Response
                # If we have tool output, prepend it to the message so Nova can summarize
                augmented_message = user_input
                if tool_msg:
                    augmented_message = f"User: {user_input}\nTool ({intent}): {tool_msg}"

                final_response = chat_engine.chat(augmented_message)

                # Step 3: Model Integrity Guard
                # Use the router to validate the final response and provide fallbacks if it's garbage.
                post_data = router.route(user_input, response=final_response)
                final_response = post_data["result"]

                # 3. Layout Rendering
                elapsed = time.time() - start_time

            # Show tool results if any
            if intent != "chat":
                _display_tool_output(intent, route_data["tool_output"], context_result=tool_msg)

            # Show Nova's response
            console.print("\n[nova]Nova:[/]")
            console.print(Panel(final_response, border_style="magenta", box=ROUNDED))

            # 4. Stats
            tokens = len(chat_engine.tokenizer.encode(final_response))
            console.print(f"[dim]Stats: {tokens} tokens | {elapsed:.2f}s response time[/dim]\n")

        except (KeyboardInterrupt, EOFError):
            console.print("\n[info]Session interrupted. Goodbye.[/]")
            break
        except Exception as e:
            console.print(f"[error]Error processing request: {e}[/]")


def _display_tool_output(intent: str, output: dict, context_result: str | None = None):
    """Specific rendering for different tool results using Rich."""
    title = f"{intent.upper()} TOOL"

    if intent == "code":
        # Check if success
        if output.get("success"):
            console.print(
                Panel(
                    output.get("output", "").strip(),
                    title="[bold green]Execution Result[/]",
                    subtitle=f"[dim]{output.get('runtime_ms', 0)}ms[/dim]",
                    border_style="green",
                )
            )
            # If code was extracted, show it with highlighting
            # (In this simple version, we assume it's in the message)
        else:
            console.print(
                Panel(
                    output.get("error", "Error"), title="[error]Code Error[/]", border_style="red"
                )
            )

    elif intent == "search":
        # Show sources from context
        if context_result:
            console.print(
                Panel(context_result, title=f"[bold yellow]{title}[/]", border_style="yellow")
            )

    elif intent == "file":
        if "content" in output:
            syntax = Syntax(output["content"][:1000], "python", theme="monokai", line_numbers=True)
            console.print(
                Panel(syntax, title=f"File: {output.get('path', 'unknown')}", border_style="blue")
            )
        else:
            console.print(
                Panel(str(context_result), title="File System Response", border_style="blue")
            )

    elif intent == "blender":
        script = output.get("script", "")
        if script:
            syntax = Syntax(script, "python", theme="monokai", line_numbers=True)
            console.print(Panel(syntax, title="Generated BPY Script", border_style="magenta"))

    elif intent == "math":
        console.print(Panel(str(context_result), title="Math Engine", border_style="green"))


if __name__ == "__main__":
    main()
