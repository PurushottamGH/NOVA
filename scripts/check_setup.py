"""
NovaMind Setup Checker
========================
Windows-friendly script that verifies the entire environment is ready for training.

Usage:
    python scripts/check_setup.py
"""

import importlib
import sys
from pathlib import Path


def check_python_version():
    """Check Python >= 3.9"""
    version = sys.version_info
    ok = version >= (3, 9)
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] Python version: {version.major}.{version.minor}.{version.micro}", end="")
    if not ok:
        print(" (need >= 3.9)")
    else:
        print()
    return ok


def check_pytorch():
    """Check PyTorch is installed and report version + CUDA."""
    try:
        import torch

        print(f"  [OK]   PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  [OK]   CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("  [WARN] CUDA not available — training will be slow on CPU")
        return True
    except ImportError:
        print("  [FAIL] PyTorch not installed — run: pip install torch")
        return False


def check_dependencies():
    """Check all required and optional packages."""
    # Core requirements
    required = [
        "numpy",
        "tqdm",
        "matplotlib",
        "requests",
        "colorama",
        "psutil",
        "sentence_transformers",
        "faiss",
        "rich",
        "prompt_toolkit",
        "bitsandbytes",
        "datasets",
        "huggingface_hub",
        "sympy",
        "chromadb",
    ]

    # Optional dependencies for NovaFileParser
    optional = {
        "fitz": "PyMuPDF (required for PDF parsing)",
        "docx": "python-docx (required for Word parsing)",
    }

    all_ok = True
    print("\n  [Core Dependencies]")
    for dep in required:
        try:
            # Special case for faiss
            mod_name = "faiss" if dep == "faiss" else dep
            importlib.import_module(mod_name)
            print(f"  [OK]   {dep}")
        except ImportError:
            print(f"  [FAIL] {dep} not installed")
            all_ok = False

    print("\n  [Optional Features]")
    for mod, desc in optional.items():
        try:
            importlib.import_module(mod)
            print(f"  [OK]   {mod} ({desc.split('(')[0].strip()})")
        except ImportError:
            print(f"  [WARN] {mod} not found — {desc}")

    return all_ok


def check_project_files():
    """Check all 28 project files exist."""
    project_root = Path(__file__).resolve().parent.parent

    required_files = [
        "model/__init__.py",
        "model/config.py",
        "model/attention.py",
        "model/positional.py",
        "model/feedforward.py",
        "model/block.py",
        "model/architecture.py",
        "model/utils.py",
        "tokenizer/__init__.py",
        "tokenizer/bpe.py",
        "tokenizer/special_tokens.py",
        "tokenizer/tokenizer.py",
        "data/__init__.py",
        "data/collector.py",
        "data/cleaner.py",
        "data/dataset.py",
        "data/dataloader.py",
        "training/__init__.py",
        "training/optimizer.py",
        "training/scheduler.py",
        "training/loss.py",
        "training/trainer.py",
        "training/checkpointing.py",
        "training/train.py",
        "inference/__init__.py",
        "inference/sampler.py",
        "inference/generate.py",
        "inference/chat.py",
        "inference/evaluate.py",
        "main.py",
        "requirements.txt",
        "setup.py",
        "test_all.py",
    ]

    missing = []
    for f in required_files:
        if not (project_root / f).exists():
            missing.append(f)

    if not missing:
        print(f"  [OK]   All {len(required_files)} project files found")
        return True
    else:
        for f in missing:
            print(f"  [FAIL] Missing: {f}")
        return False


def check_data_dir():
    """Check personal_data/ has at least one .txt file."""
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "personal_data"

    if not data_dir.exists():
        print("  [WARN] personal_data/ directory not found — run: python -m data.collector")
        return False

    txt_files = list(data_dir.glob("*.txt"))
    if not txt_files:
        print("  [WARN] No .txt files in personal_data/ — run: python -m data.collector")
        return False

    total_chars = sum(f.stat().st_size for f in txt_files)
    print(f"  [OK]   personal_data/ has {len(txt_files)} .txt files ({total_chars:,} bytes)")
    return True


def check_directories():
    """Check weights/ and logs/ exist (create if not)."""
    project_root = Path(__file__).resolve().parent.parent
    all_ok = True

    for dirname in ["weights", "logs"]:
        dirpath = project_root / dirname
        if dirpath.exists():
            print(f"  [OK]   {dirname}/ directory exists")
        else:
            dirpath.mkdir(parents=True, exist_ok=True)
            print(f"  [OK]   {dirname}/ created")

    return all_ok


def main():
    print("=" * 55)
    print("  NovaMind Setup Checker")
    print("  Built by Purushottam")
    print("=" * 55)

    results = {}

    print("\n[1] Python Version:")
    results["python"] = check_python_version()

    print("\n[2] PyTorch:")
    results["pytorch"] = check_pytorch()

    print("\n[3] Dependencies:")
    results["deps"] = check_dependencies()

    print("\n[4] Project Files:")
    results["files"] = check_project_files()

    print("\n[5] Training Data:")
    results["data"] = check_data_dir()

    print("\n[6] Output Directories:")
    results["dirs"] = check_directories()

    # Final verdict
    print("\n" + "=" * 55)
    critical_ok = results["python"] and results["pytorch"] and results["files"]
    all_ok = all(results.values())

    if all_ok:
        print("  [READY] NovaMind is fully set up!")
    elif critical_ok:
        print("  [WARN]  MOSTLY READY — some optional items missing (see WARN above)")
        print("          You can still train, but run data collection first.")
    else:
        print("  [ERROR] NOT READY — fix the FAIL items above before training")
    print("=" * 55)


if __name__ == "__main__":
    main()
