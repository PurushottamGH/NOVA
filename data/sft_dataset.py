"""
NovaMind SFT Dataset Utilities
================================
Formatting logic for Supervised Fine-Tuning (SFT).
Standardizes instruction-following pairs into the model's preferred chat format.
"""

def format_instruction(instruction, response):
    """
    Format an instruction-response pair into a chat-like string.
    
    Args:
        instruction: The task or question from the user.
        response: The desired answer from the assistant.
        
    Returns:
        Formatted multi-line string.
    """
    return (
        f"<|user|>\n{instruction}\n"
        f"<|assistant|>\n{response}\n"
    )
