"""
NovaMind Chat UI — Premium Terminal Interface
===============================================
A beautiful, Claude Code-inspired chat interface for Nova.
"""

import sys
import time
import argparse
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

# Rich for UI rendering
from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.columns import Columns
from rich.markdown import Markdown
from rich.theme import Theme
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table

# Prompt Toolkit for input
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML

# Nova logic
from model.config import NovaMindConfig
from model.architecture import NovaMind
from tokenizer.tokenizer import NovaMindTokenizer
from inference.generate import generate_text

# Setup Rich Console
custom_theme = Theme({
    "nova.name": "bold white",
    "nova.version": "dim white",
    "nova.model": "dim white",
    "nova.path": "dim",
    "nova.logo": "bold #e06c75", # Salmon pink
    "nova.bullet": "bold white",
})
console = Console(theme=custom_theme)

# Setup Prompt Toolkit
prompt_style = Style.from_dict({
    'prompt': 'bg:#444444 fg:#ffffff bold',
    'bottom-toolbar': 'fg:#aaaaaa bg:#222222',
    'line': 'fg:#444444',
})

def bottom_toolbar():
    return HTML(' <b>?</b> for shortcuts <style fg="#444444">|</style> <style fg="white">Thinking off</style> (type /think to toggle) ')

# Optimizations
torch.set_num_threads(torch.get_num_threads())
torch.set_num_interop_threads(2)

def optimize_model(model):
    model.eval()
    device = next(model.parameters()).device
    if device.type == "cpu":
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    else:
        model = model.half()

    try:
        if hasattr(torch, "compile"):
            compiled_model = torch.compile(model, mode="reduce-overhead")
            with torch.inference_mode():
                dummy_input = torch.zeros(1, 1, dtype=torch.long, device=device)
                _ = compiled_model(dummy_input)
            return compiled_model
    except Exception:
        pass
    return model

SYSTEM_PROMPT = (
    "You are Nova, an intelligent personal AI assistant built for Purushottam. "
    "You are knowledgeable in AI, space, astronomy, data science, and software engineering."
)

def format_prompt(user_message: str, history: list) -> str:
    parts = [f"<|system|>\n{SYSTEM_PROMPT}\n"]
    for turn in history:
        parts.append(f"<|user|>\n{turn['user']}\n")
        parts.append(f"<|assistant|>\n{turn['nova']}\n")
    parts.append(f"<|user|>\n{user_message}\n")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


def print_header(model_name="weights/best.pt", version="v2.0.24"):
    logo = """
 ▄▄        ▄▄ 
██████████████
██  ██  ██  ██
██████████████
   ██    ██   
"""
    logo_text = Text(logo.strip("\n"), style="nova.logo")
    
    info_text = Text()
    info_text.append("Nova Code ", style="nova.name")
    info_text.append(version, style="nova.version")
    info_text.append("\n")
    info_text.append("Model • ", style="dim")
    info_text.append(model_name, style="nova.model")
    info_text.append("\n")
    info_text.append(os.getcwd(), style="nova.path")
    
    col = Columns([logo_text, info_text], expand=False, padding=(0, 2))
    
    console.print()
    console.print(col)
    console.print()


def print_nova_message(response):
    console.print()
    table = Table(show_header=False, show_edge=False, box=None, padding=(0, 1))
    table.add_column("Bullet", style="nova.bullet", no_wrap=True)
    table.add_column("Message", overflow="fold")
    table.add_row("●", Markdown(response))
    console.print(table)
    console.print()


def main():
    parser = argparse.ArgumentParser(description="Nova Chat UI")
    parser.add_argument("--model_path", type=str, default="weights/final_model",
                        help="Path to saved model directory")
    parser.add_argument("--tokenizer_path", type=str, default="weights/tokenizer",
                        help="Path to saved tokenizer directory")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto/cuda/cpu")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    tokenizer_path = Path(args.tokenizer_path)

    console.clear()

    with console.status("[dim]Waking up Nova...[/dim]", spinner="dots"):
        # Load Tokenizer
        if (tokenizer_path / "vocab.json").exists():
            tokenizer = NovaMindTokenizer.load(str(tokenizer_path))
        else:
            tokenizer = NovaMindTokenizer()
            temp_file = Path("scripts/_temp_chat.txt")
            temp_file.write_text("Hello Nova", encoding="utf-8")
            tokenizer.train([str(temp_file)], vocab_size=200)
            temp_file.unlink(missing_ok=True)

        # Load Model
        config_path = model_path / "config.json"
        if config_path.exists():
            model = NovaMind.load(str(model_path), device=args.device)
            m_name = f"NovaMind • {model_path.name}"
        else:
            config = NovaMindConfig(vocab_size=tokenizer.vocab_size, device=args.device)
            model = NovaMind(config)
            model.to(config.device)
            m_name = "NovaMind (Untrained Demo)"

        model = optimize_model(model)

    console.clear()
    print_header(model_name=m_name)

    history = []
    session = PromptSession(style=prompt_style)

    while True:
        console.rule(style="dim #444444")
        try:
            # We add a little space before the input prompt to match the design
            user_input = session.prompt(HTML('<style bg="#444444" fg="#ffffff"> &gt; </style> '), bottom_toolbar=bottom_toolbar)
            user_input = user_input.strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Nova is going to sleep. Goodbye![/dim]")
            break

        if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
            console.print("\n[dim]Nova is going to sleep. Goodbye![/dim]")
            break
            
        if user_input.lower() == "/think":
            console.print("\n[dim italic]Thinking mode toggled (mock functionality).[/dim]\n")
            continue

        if not user_input:
            continue

        # Format input and generate
        prompt = format_prompt(user_input, history)

        nova_response = ""
        with console.status("[dim white]Nova is typing...[/dim white]", spinner="dots", spinner_style="white"):
            try:
                response = generate_text(
                    model, tokenizer, prompt,
                    max_new_tokens=200,
                    temperature=0.8, top_k=50, top_p=0.9,
                    repetition_penalty=1.15,
                )
                
                nova_response = response
                if "<|assistant|>" in nova_response:
                    parts = nova_response.rsplit("<|assistant|>", 1)
                    if len(parts) > 1:
                        nova_response = parts[1].strip()

                if "<|user|>" in nova_response:
                    nova_response = nova_response.split("<|user|>")[0].strip()

                if not nova_response:
                    nova_response = "I need more training data to answer that."
            except Exception as e:
                nova_response = f"**Error details:** `{e}`"

        print_nova_message(nova_response)
        history.append({"user": user_input, "nova": nova_response})


if __name__ == "__main__":
    main()
