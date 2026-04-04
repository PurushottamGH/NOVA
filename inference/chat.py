"""
NovaMind Chat Engine
======================
Conversational wrapper for the NovaMind model that serves as a drop-in
replacement for the Groq API in the Nova FastAPI backend.

The NovaChatEngine manages conversation history, formats prompts,
and provides both blocking and streaming chat interfaces.

Drop-in replacement for Groq:
    # Before (Groq):
    # response = groq_client.chat.completions.create(messages=messages)
    
    # After (NovaMind):
    engine = NovaChatEngine("weights/final_model", "weights/tokenizer")
    response = engine.chat("Hello!", history=[])

System prompt is hardcoded for Nova's personality.
"""

import torch
from pathlib import Path
from typing import List, Dict, Generator

# 1. Use all CPU cores for maximum inference speed
torch.set_num_threads(torch.get_num_threads())
torch.set_num_interop_threads(2)

# 2. Dynamic Quantization + 3. Compilation
def optimize_model(model):
    model.eval()
    device = next(model.parameters()).device
    
    # 1. Quantization only works on CPU
    if device.type == "cpu":
        print("[NovaChatEngine] Applying dynamic quantization for CPU speedup...")
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    else:
        # On GPU, we use Half Precision (FP16) for a 2x speed boost instead
        print(f"[NovaChatEngine] Using FP16 acceleration on {device}")
        model = model.half()

    base_model = model
    try:
        if hasattr(torch, "compile"):
            # Note: compilation is often not available on Windows
            compiled_model = torch.compile(model, mode="reduce-overhead")
            # Force evaluation to trigger lazy compilation immediately
            with torch.inference_mode():
                dummy_input = torch.zeros(1, 1, dtype=torch.long, device=device)
                _ = compiled_model(dummy_input)
            print("[NovaChatEngine] Model compiled for faster inference")
            return compiled_model
        return base_model
    except Exception as e:
        print(f"[NovaChatEngine] Running without compile (Compiler not available: {type(e).__name__})")
        return base_model

from model.architecture import NovaMind
from model.config import NovaMindConfig
from tokenizer.tokenizer import NovaMindTokenizer
from inference.generate import stream_generate, generate_text
from nova_modules.math_engine import NovaMathEngine  # FIXED: added for symbolic math support


# Nova's system prompt — defines the AI personality
NOVA_SYSTEM_PROMPT = (
    "You are Nova, an intelligent personal AI assistant built for Purushottam. "
    "You are knowledgeable in AI, space, astronomy, data science, and software engineering. "
    "You give direct, complete, intelligent answers. You remember context."
)


class NovaChatEngine:
    """
    Conversational AI engine for Nova — drop-in replacement for Groq API.
    
    Usage in FastAPI:
        engine = NovaChatEngine("weights/final_model", "weights/tokenizer")
        
        @app.post("/chat")
        async def chat(request: ChatRequest):
            response = engine.chat(request.message, request.history)
            return {"response": response}
    
    Args:
        model_path: Path to saved NovaMind model directory
        tokenizer_path: Path to saved tokenizer directory
        device: Device to run inference on ("auto", "cuda", "cpu", "mps")
        max_response_tokens: Maximum tokens in generated response
    """

    def __init__(
        self,
        model_path: str = "weights/final_model",
        tokenizer_path: str = "weights/tokenizer",
        device: str = "auto",
        max_response_tokens: int = 300,
    ):
        # Resolve device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.max_response_tokens = max_response_tokens

        # Load model
        print(f"[NovaChatEngine] Loading model from {model_path}...")
        self.model = NovaMind.load(model_path, device=device)
        self.model = optimize_model(self.model)
        self.model.eval()  # Call model.eval() after loading and optimization

        # Load tokenizer
        print(f"[NovaChatEngine] Loading tokenizer from {tokenizer_path}...")
        self.tokenizer = NovaMindTokenizer.load(tokenizer_path)

        # Conversation history
        self._history: List[Dict[str, str]] = []

        # Specialized Engines
        self.math_engine = NovaMathEngine()  # FIXED: added to handle exact math tasks

        from nova_modules.memory import NovaMemory
        self.memory = NovaMemory()

        print(f"[NovaChatEngine] Ready on {device}")

    def _format_prompt(self, user_message: str, history: List[Dict[str, str]] = None) -> str:
        """
        Format conversation history + new message into a prompt string.
        
        FIXED: System prompt is ALWAYS first. Context overflow drops oldest turns.
        
        Format:
            System: {NOVA_SYSTEM_PROMPT}
            User: {message_1}
            Nova: {response_1}
            User: {message_2}
            Nova:
        
        Args:
            user_message: The new user message
            history: List of {"role": "user"/"nova", "content": "..."} dicts
        
        Returns:
            Formatted prompt string
        """
        if history is None:
            history = []
        
        # Limiting history to last 6 turns (3 pairs) for focused context
        if len(history) > 6:
            history = history[-6:]

        # FIXED: always keep system prompt first — this is the anchor
        parts = [f"<|system|>\n{NOVA_SYSTEM_PROMPT}\n"]

        # Add conversation history
        for turn in history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role == "user":
                parts.append(f"<|user|>\n{content}\n")
            elif role in ("nova", "assistant"):
                parts.append(f"<|assistant|>\n{content}\n")

        # Add the new user message and prompt for Nova's response
        parts.append(f"<|user|>\n{user_message}\n")
        parts.append("<|assistant|>\n")

        prompt = "\n".join(parts)

        # FIXED: Context window overflow handling — drop oldest turns first
        # Keep system prompt + latest user message, trim old history
        context_limit = self.model.config.context_length - 100  # Leave room for response
        while len(self.tokenizer.encode(prompt)) > context_limit and len(history) > 0:
            # Drop the oldest turn (first 2 entries = one user + one nova turn)
            if len(history) >= 2:
                history = history[2:]  # Drop oldest user+nova pair
            elif len(history) >= 1:
                history = history[1:]  # Drop oldest single turn
            else:
                break

            # Rebuild prompt with reduced history
            parts = [f"<|system|>\n{NOVA_SYSTEM_PROMPT}\n"]
            for turn in history:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role == "user":
                    parts.append(f"<|user|>\n{content}\n")
                elif role in ("nova", "assistant"):
                    parts.append(f"<|assistant|>\n{content}\n")
            parts.append(f"<|user|>\n{user_message}\n")
            parts.append("<|assistant|>\n")
            prompt = "\n".join(parts)

        return prompt

    def chat(self, user_message: str, history: List[Dict[str, str]] = None) -> str:
        """
        Generate a response to the user's message.
        
        This is the main chat interface — a drop-in replacement for Groq API.
        
        Args:
            user_message: The user's message text
            history: Conversation history as list of {"role": ..., "content": ...}
                     If None, uses internal history
        
        Returns:
            Nova's response text
        """
        if history is None:
            history = self._history

        # FIXED: Check math engine first — exact answers beat LLM guesses
        math_result = self.math_engine.detect_and_solve(user_message)
        processed_message = user_message
        if math_result:
            # Still pass through NovaMind for explanation/context
            print(f"  [Chat] Math detected: {math_result}")
            processed_message = (
                f"{user_message}\n"
                f"[Math Engine Result: {math_result}]"
            )

        # Inject memory context
        memory_ctx = self.memory.inject_memory(processed_message)
        if memory_ctx:
            processed_message = f"{memory_ctx}\n{processed_message}"

        # Format the full prompt
        prompt = self._format_prompt(processed_message, history)

        # FIXED: temperature guard — clamp to safe range
        temperature = 0.7  # Base temperature update
        temperature = max(0.1, min(temperature, 2.0))  # Clamp to safe range

        # FIXED: empty response guard with retry — up to 3 attempts with increasing temperature
        nova_response = ""
        for attempt in range(3):
            response = generate_text(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_new_tokens=self.max_response_tokens,
                temperature=temperature,
                top_k=40,
                top_p=0.9,
                repetition_penalty=1.3,
            )

            # Extract only Nova's response (after the last "<|assistant|>\n" in the output)
            nova_response = response
            if "<|assistant|>" in nova_response:
                parts = nova_response.rsplit("<|assistant|>", 1)
                if len(parts) > 1:
                    nova_response = parts[1].strip()

            # Clean up: stop at "<|user|>" if the model continues generating
            if "<|user|>" in nova_response:
                nova_response = nova_response.split("<|user|>")[0].strip()

            # FIXED: check if response is actually meaningful
            if nova_response.strip() and nova_response.strip() not in ("", "[EOS]", "[PAD]", "<EOS>", "<PAD>"):
                break  # Got a valid response

            # Retry with higher temperature
            temperature = min(temperature + 0.3, 2.0)  # FIXED: increase randomness on retry
            print(f"  [Chat] Empty response, retrying with temperature={temperature:.1f} (attempt {attempt + 1}/3)")
        else:
            # FIXED: all retries failed — return a fallback message
            nova_response = "I need more training data to answer that."

        # Store in persistent memory
        self.memory.remember(user_message, nova_response)

        # Update internal history
        self._history.append({"role": "user", "content": user_message})
        self._history.append({"role": "nova", "content": nova_response})

        return nova_response

    def stream_chat(
        self, user_message: str, history: List[Dict[str, str]] = None
    ) -> Generator[str, None, None]:
        """
        Stream a response token by token (generator interface).
        
        Yields individual tokens as they are generated, enabling
        real-time streaming in a FastAPI endpoint.
        
        Args:
            user_message: The user's message
            history: Conversation history
        
        Yields:
            Individual token strings as they are generated
        """
        if history is None:
            history = self._history

        prompt = self._format_prompt(user_message, history)
        full_response = []

        for token_text in stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_new_tokens=self.max_response_tokens,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.3,
        ):
            # Stop if the model starts generating "<|user|>" (end of response)
            full_so_far = "".join(full_response) + token_text
            if "<|user|>" in full_so_far:
                break
            full_response.append(token_text)
            yield token_text

        # Update history with the complete response
        complete_response = "".join(full_response).strip()

        # FIXED: empty response guard for streaming
        if not complete_response:
            complete_response = "I need more training data to answer that."
            yield complete_response

        self._history.append({"role": "user", "content": user_message})
        self._history.append({"role": "nova", "content": complete_response})

    def reset_history(self):
        """Clear the conversation history."""
        self._history = []
        print("[NovaChatEngine] History cleared")

    @property
    def history(self) -> List[Dict[str, str]]:
        """Get current conversation history."""
        return self._history.copy()
