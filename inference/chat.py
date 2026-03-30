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
from typing import List, Dict, Generator
from pathlib import Path

from model.architecture import NovaMind
from model.config import NovaMindConfig
from tokenizer.tokenizer import NovaMindTokenizer
from inference.generate import stream_generate, generate_text


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
        model_path: str,
        tokenizer_path: str,
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
        self.model.eval()

        # Load tokenizer
        print(f"[NovaChatEngine] Loading tokenizer from {tokenizer_path}...")
        self.tokenizer = NovaMindTokenizer.load(tokenizer_path)

        # Conversation history
        self._history: List[Dict[str, str]] = []

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

        # FIXED: always keep system prompt first — this is the anchor
        parts = [f"System: {NOVA_SYSTEM_PROMPT}\n"]

        # Add conversation history
        for turn in history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role == "user":
                parts.append(f"User: {content}")
            elif role in ("nova", "assistant"):
                parts.append(f"Nova: {content}")

        # Add the new user message and prompt for Nova's response
        parts.append(f"User: {user_message}")
        parts.append("Nova:")

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
            parts = [f"System: {NOVA_SYSTEM_PROMPT}\n"]
            for turn in history:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role == "user":
                    parts.append(f"User: {content}")
                elif role in ("nova", "assistant"):
                    parts.append(f"Nova: {content}")
            parts.append(f"User: {user_message}")
            parts.append("Nova:")
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

        # Format the full prompt
        prompt = self._format_prompt(user_message, history)

        # FIXED: temperature guard — clamp to safe range
        temperature = 0.8  # Base temperature
        temperature = max(0.1, min(temperature, 2.0))  # FIXED: guard against bad values

        # FIXED: empty response guard with retry — up to 3 attempts with increasing temperature
        nova_response = ""
        for attempt in range(3):
            response = generate_text(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_new_tokens=self.max_response_tokens,
                temperature=temperature,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.15,
            )

            # Extract only Nova's response (after the last "Nova:" in the output)
            nova_response = response
            if "Nova:" in nova_response:
                parts = nova_response.rsplit("Nova:", 1)
                if len(parts) > 1:
                    nova_response = parts[1].strip()

            # Clean up: stop at "User:" if the model continues generating
            if "User:" in nova_response:
                nova_response = nova_response.split("User:")[0].strip()

            # FIXED: check if response is actually meaningful
            if nova_response.strip() and nova_response.strip() not in ("", "[EOS]", "[PAD]", "<EOS>", "<PAD>"):
                break  # Got a valid response

            # Retry with higher temperature
            temperature = min(temperature + 0.3, 2.0)  # FIXED: increase randomness on retry
            print(f"  [Chat] Empty response, retrying with temperature={temperature:.1f} (attempt {attempt + 1}/3)")
        else:
            # FIXED: all retries failed — return a fallback message
            nova_response = "I need more training data to answer that."

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
            temperature=max(0.1, min(0.8, 2.0)),  # FIXED: temperature guard
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.15,
        ):
            # Stop if the model starts generating "User:" (end of response)
            full_so_far = "".join(full_response) + token_text
            if "User:" in full_so_far:
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
