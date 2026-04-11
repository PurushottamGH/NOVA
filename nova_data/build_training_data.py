"""
Nova Training Data Pipeline
==============================
Downloads, formats, and merges multiple HuggingFace datasets into
a single JSONL training corpus for NovaMind fine-tuning.

Datasets:
    - hendrycks/competition_math (competition math reasoning)
    - gsm8k (grade school math word problems)
    - openai/openai_humaneval (code problems)
    - codeparrot/apps (code challenges)
    - TIGER-Lab/MathInstruct (mixed math instruction)

Output: nova_data/training_corpus.jsonl

Usage:
    python nova_data/build_training_data.py
"""

import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

# Maximum samples per dataset to avoid memory issues
MAX_SAMPLES_PER_DATASET = 50_000

# Output directory and file
OUTPUT_DIR = Path(__file__).parent
OUTPUT_FILE = OUTPUT_DIR / "training_corpus.jsonl"


# ====================================================================== #
#  Format functions — one per dataset
# ====================================================================== #


def format_competition_math(sample: dict) -> dict | None:
    """
    Format a sample from hendrycks/competition_math.

    Args:
        sample: Raw dataset sample with 'problem' and 'solution' fields.

    Returns:
        Formatted dict with 'text' key, or None if fields are empty.
    """
    try:
        question = sample.get("problem", "")
        solution = sample.get("solution", "")
        if not question or not solution:
            return None
        sol_clean = solution.strip()
        boxed_marker = "\\boxed"
        answer_part = sol_clean.split(boxed_marker)[-1] if boxed_marker in solution else sol_clean
        text = f"User: {question.strip()}\n\nNova: {sol_clean}\n\nAnswer: {answer_part}"
        return {"text": text}
    except Exception:
        return None


def format_gsm8k(sample: dict) -> dict | None:
    """
    Format a sample from gsm8k.

    Args:
        sample: Raw dataset sample with 'question' and 'answer' fields.

    Returns:
        Formatted dict with 'text' key, or None if fields are empty.
    """
    try:
        question = sample.get("question", "")
        answer = sample.get("answer", "")
        if not question or not answer:
            return None
        # GSM8K answers have step-by-step reasoning ending with ####
        text = (
            f"User: {question.strip()}\n\n"
            f"Nova: {answer.strip()}\n\n"
            f"Answer: {answer.strip().split('####')[-1].strip() if '####' in answer else answer.strip()}"
        )
        return {"text": text}
    except Exception:
        return None


def format_humaneval(sample: dict) -> dict | None:
    """
    Format a sample from openai/openai_humaneval.

    Args:
        sample: Raw dataset sample with 'prompt' and 'canonical_solution' fields.

    Returns:
        Formatted dict with 'text' key, or None if fields are empty.
    """
    try:
        prompt = sample.get("prompt", "")
        solution = sample.get("canonical_solution", "")
        if not prompt or not solution:
            return None
        text = (
            f"User: Complete the following Python function:\n{prompt.strip()}\n\n"
            f"Nova: Here's the implementation:\n{solution.strip()}\n\n"
            f"Answer: {solution.strip()}"
        )
        return {"text": text}
    except Exception:
        return None


def format_apps(sample: dict) -> dict | None:
    """
    Format a sample from codeparrot/apps.

    Args:
        sample: Raw dataset sample with 'question' and 'solutions' fields.

    Returns:
        Formatted dict with 'text' key, or None if fields are empty.
    """
    try:
        question = sample.get("question", "")
        solutions = sample.get("solutions", "")
        if not question or not solutions:
            return None
        # solutions is a JSON string of list of solutions
        try:
            sol_list = json.loads(solutions)
            if isinstance(sol_list, list) and len(sol_list) > 0:
                solution = sol_list[0]
            else:
                solution = str(solutions)
        except (json.JSONDecodeError, TypeError):
            solution = str(solutions)
        if not solution:
            return None
        text = (
            f"User: {question.strip()}\n\n"
            f"Nova: Here's a solution:\n{str(solution).strip()}\n\n"
            f"Answer: {str(solution).strip()}"
        )
        return {"text": text}
    except Exception:
        return None


def format_mathinstruct(sample: dict) -> dict | None:
    """
    Format a sample from TIGER-Lab/MathInstruct.

    Args:
        sample: Raw dataset sample with 'instruction' and 'output' fields.

    Returns:
        Formatted dict with 'text' key, or None if fields are empty.
    """
    try:
        instruction = sample.get("instruction", "")
        output = sample.get("output", "")
        if not instruction or not output:
            return None
        text = f"User: {instruction.strip()}\n\nNova: {output.strip()}\n\nAnswer: {output.strip()}"
        return {"text": text}
    except Exception:
        return None


# ====================================================================== #
#  Deduplication
# ====================================================================== #


def deduplicate(samples: list[dict]) -> list[dict]:
    """
    Remove exact duplicate 'text' fields from the samples list.

    Args:
        samples: List of dicts with 'text' keys.

    Returns:
        Deduplicated list preserving original order.
    """
    seen = set()
    unique = []
    for sample in samples:
        text = sample.get("text", "")
        if text not in seen:
            seen.add(text)
            unique.append(sample)
    return unique


# ====================================================================== #
#  Dataset loading and processing
# ====================================================================== #


def load_and_format_dataset(
    name: str, config: str | None, split: str, formatter, label: str
) -> list[dict]:
    """
    Load a HuggingFace dataset and format all samples.

    Args:
        name: HuggingFace dataset name.
        config: Dataset configuration (or None).
        split: Dataset split to load.
        formatter: Function to format each sample.
        label: Human-readable label for progress display.

    Returns:
        List of formatted samples.
    """
    try:
        print(f"\n📥 Loading: {label} ({name})...")
        if config:
            dataset = load_dataset(name, config, split=split, trust_remote_code=True)
        else:
            dataset = load_dataset(name, split=split, trust_remote_code=True)

        # Cap at MAX_SAMPLES_PER_DATASET
        total = min(len(dataset), MAX_SAMPLES_PER_DATASET)
        formatted = []

        for i in tqdm(range(total), desc=f"Formatting {label}"):
            result = formatter(dataset[i])
            if result is not None:
                formatted.append(result)

        print(f"  ✅ {label}: {len(formatted)} valid samples (from {total} raw)")
        return formatted

    except Exception as e:
        print(f"  ❌ Failed to load {label}: {e}")
        return []


def main():
    """Main pipeline: download, format, merge, deduplicate, and save."""
    print("=" * 60)
    print("  NovaMind Training Data Pipeline")
    print("=" * 60)

    all_samples = []
    stats = {}

    # --- Dataset 1: Competition Math ---
    samples = load_and_format_dataset(
        name="hendrycks/competition_math",
        config=None,
        split="train",
        formatter=format_competition_math,
        label="Competition Math",
    )
    stats["Competition Math"] = len(samples)
    all_samples.extend(samples)

    # --- Dataset 2: GSM8K ---
    samples = load_and_format_dataset(
        name="gsm8k",
        config="main",
        split="train",
        formatter=format_gsm8k,
        label="GSM8K",
    )
    stats["GSM8K"] = len(samples)
    all_samples.extend(samples)

    # --- Dataset 3: HumanEval ---
    samples = load_and_format_dataset(
        name="openai/openai_humaneval",
        config=None,
        split="test",
        formatter=format_humaneval,
        label="HumanEval",
    )
    stats["HumanEval"] = len(samples)
    all_samples.extend(samples)

    # --- Dataset 4: APPS ---
    samples = load_and_format_dataset(
        name="codeparrot/apps",
        config="all",
        split="train",
        formatter=format_apps,
        label="APPS",
    )
    stats["APPS"] = len(samples)
    all_samples.extend(samples)

    # --- Dataset 5: MathInstruct ---
    samples = load_and_format_dataset(
        name="TIGER-Lab/MathInstruct",
        config=None,
        split="train",
        formatter=format_mathinstruct,
        label="MathInstruct",
    )
    stats["MathInstruct"] = len(samples)
    all_samples.extend(samples)

    # --- Deduplicate ---
    print(f"\n🔄 Deduplicating {len(all_samples)} samples...")
    all_samples = deduplicate(all_samples)
    print(f"  ✅ After dedup: {len(all_samples)} unique samples")

    # --- Write output ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n💾 Writing to {OUTPUT_FILE}...")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for sample in tqdm(all_samples, desc="Writing JSONL"):
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # --- Print stats ---
    total_chars = sum(len(s["text"]) for s in all_samples)
    est_tokens = total_chars // 4

    print("\n" + "=" * 60)
    print("  📊 Pipeline Summary")
    print("=" * 60)
    for name, count in stats.items():
        print(f"  {name:<25} {count:>8,} samples")
    print(f"  {'─' * 35}")
    print(f"  {'Total (pre-dedup)':<25} {sum(stats.values()):>8,} samples")
    print(f"  {'Total (post-dedup)':<25} {len(all_samples):>8,} samples")
    print(f"  {'Estimated tokens':<25} {est_tokens:>8,}")
    print(f"  {'Output file':<25} {OUTPUT_FILE}")
    print("=" * 60)
    print("✅ Done!")


if __name__ == "__main__":
    main()
