"""
NovaMind Text Cleaner
======================
Cleans and normalizes raw text data for training.

Processing steps:
1. Unicode normalization (NFKD → ASCII-compatible)
2. Remove non-printable and control characters
3. Normalize whitespace (collapse multiple spaces/newlines)
4. Remove very short lines (likely headers/footers)
5. Remove duplicate lines
6. Basic sentence boundary preservation

Usage:
    python -m data.cleaner --input personal_data/ --output personal_data/cleaned/
"""

import re
import unicodedata
from pathlib import Path


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode characters to their closest ASCII equivalents.
    Handles smart quotes, em-dashes, ligatures, etc.
    """
    # Normalize to NFKD form (decompose ligatures, etc.)
    text = unicodedata.normalize("NFKD", text)

    # Replace common special characters
    replacements = {
        "\u2018": "'",
        "\u2019": "'",  # Smart single quotes → apostrophe
        "\u201c": '"',
        "\u201d": '"',  # Smart double quotes → double quote
        "\u2014": " — ",
        "\u2013": " - ",  # Em/en dashes
        "\u2026": "...",  # Ellipsis
        "\u00a0": " ",  # Non-breaking space
        "\u200b": "",  # Zero-width space
        "\ufeff": "",  # BOM
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def remove_control_chars(text: str) -> str:
    """Remove non-printable control characters except newlines and tabs."""
    cleaned = []
    for char in text:
        if char in ("\n", "\t", "\r"):
            cleaned.append(char)
        elif unicodedata.category(char) in ("Cc", "Cf"):
            continue  # Skip control and format characters
        else:
            cleaned.append(char)
    return "".join(cleaned)


def normalize_whitespace(text: str) -> str:
    """
    Collapse multiple whitespace while preserving paragraph breaks.
    - Multiple spaces → single space
    - 3+ newlines → double newline (paragraph break)
    - Trim trailing whitespace on each line
    """
    # Replace tabs with spaces
    text = text.replace("\t", "    ")

    # Collapse multiple spaces on the same line
    text = re.sub(r"[^\S\n]+", " ", text)

    # Collapse 3+ consecutive newlines to 2 (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Trim trailing whitespace on each line
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text


def remove_short_lines(text: str, min_length: int = 10) -> str:
    """
    Remove lines that are too short to be meaningful content.
    These are usually headers, page numbers, or formatting artifacts.
    """
    lines = text.split("\n")
    filtered = []
    for line in lines:
        stripped = line.strip()
        if len(stripped) == 0:
            filtered.append("")  # Keep empty lines for paragraph breaks
        elif len(stripped) >= min_length:
            filtered.append(line)
        # Skip lines with < min_length non-empty chars (noise)
    return "\n".join(filtered)


def remove_duplicate_lines(text: str) -> str:
    """Remove consecutive duplicate lines."""
    lines = text.split("\n")
    if not lines:
        return text

    filtered = [lines[0]]
    for line in lines[1:]:
        if line.strip() != filtered[-1].strip() or line.strip() == "":
            filtered.append(line)
    return "\n".join(filtered)


def clean_text(text: str, min_line_length: int = 10) -> str:
    """
    Apply the full cleaning pipeline to a text string.

    Pipeline:
    1. Unicode normalization
    2. Remove control characters
    3. Normalize whitespace
    4. Remove short lines
    5. Remove duplicate lines
    6. Final strip

    Args:
        text: Raw input text
        min_line_length: Minimum line length to keep

    Returns:
        Cleaned text string
    """
    text = normalize_unicode(text)
    text = remove_control_chars(text)
    text = normalize_whitespace(text)
    text = remove_short_lines(text, min_length=min_line_length)
    text = remove_duplicate_lines(text)
    text = text.strip()
    return text


def clean_file(input_path: str, output_path: str | None = None) -> str:
    """
    Clean a single text file.

    Args:
        input_path: Path to the raw text file
        output_path: Path to save cleaned text (if None, overwrites input)

    Returns:
        Cleaned text
    """
    path = Path(input_path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    original_len = len(text)

    cleaned = clean_text(text)
    cleaned_len = len(cleaned)

    if output_path is None:
        output_path = input_path

    Path(output_path).write_text(cleaned, encoding="utf-8")

    reduction = (1 - cleaned_len / original_len) * 100 if original_len > 0 else 0
    print(
        f"  Cleaned {path.name}: {original_len:,} → {cleaned_len:,} chars ({reduction:.1f}% reduction)"
    )

    return cleaned


def clean_directory(input_dir: str, output_dir: str | None = None):
    """
    Clean all .txt files in a directory.

    Args:
        input_dir: Directory with raw text files
        output_dir: Directory for cleaned files (if None, creates 'cleaned/' subdir)
    """
    input_path = Path(input_dir)
    output_path = input_path / "cleaned" if output_dir is None else Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    txt_files = list(input_path.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return

    print(f"Cleaning {len(txt_files)} files from {input_dir} → {output_path}")

    total_before = 0
    total_after = 0

    for txt_file in sorted(txt_files):
        text = txt_file.read_text(encoding="utf-8", errors="ignore")
        total_before += len(text)

        cleaned = clean_text(text)
        total_after += len(cleaned)

        out_file = output_path / txt_file.name
        out_file.write_text(cleaned, encoding="utf-8")
        print(f"  ✓ {txt_file.name}")

    reduction = (1 - total_after / total_before) * 100 if total_before > 0 else 0
    print(f"\nTotal: {total_before:,} → {total_after:,} chars ({reduction:.1f}% reduction)")


if __name__ == "__main__":
    import sys

    input_dir = sys.argv[1] if len(sys.argv) > 1 else "personal_data"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    clean_directory(input_dir, output_dir)
