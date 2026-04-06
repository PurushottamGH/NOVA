import re
import zipfile
from pathlib import Path


def sanitize_filename(filename):
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    # Replace anything not alphanumeric or underscore with underscore
    clean_stem = re.sub(r"[^a-zA-Z0-9]", "_", stem)
    clean_stem = re.sub(r"_+", "_", clean_stem).strip("_")
    return f"{clean_stem.lower()}{suffix}"


def create_clean_zip(output_path, root_dir, target_dirs, target_files):
    added_paths = set()

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # 1. Handle explicit directories
        for t_dir in target_dirs:
            p = Path(root_dir) / t_dir
            if not p.exists():
                print(f"Skipping {t_dir} (does not exist)")
                continue

            for file in p.rglob("*"):
                if file.is_file():
                    # Sanitize the filename if it's in personal_data
                    if t_dir == "personal_data":
                        clean_name = sanitize_filename(file.name)
                        arcname = f"personal_data/{clean_name}"
                    else:
                        # Standard path with forward slashes
                        arcname = str(file.relative_to(Path(root_dir))).replace("\\", "/")

                    if arcname not in added_paths:
                        zipf.write(file, arcname)
                        added_paths.add(arcname)
                    else:
                        print(f"Skipping duplicate: {arcname}")

        # 2. Handle explicit files
        for t_file in target_files:
            p = Path(root_dir) / t_file
            if p.exists() and p.is_file():
                arcname = p.name
                if arcname not in added_paths:
                    zipf.write(p, arcname)
                    added_paths.add(arcname)


if __name__ == "__main__":
    project_root = r"C:\Users\DELL\OneDrive\Documents\NOVA"
    output_zip = r"C:\Users\DELL\Downloads\NovaMind_Kaggle_SFT_Final_v2.zip"

    required_dirs = [
        "personal_data",
        "weights",
        "training",
        "model",
        "tokenizer",
        "data",
        "nova_modules",
    ]
    required_files = ["requirements.txt", "setup.py"]

    print("Creating clean, non-duplicate ZIP for Kaggle...")
    create_clean_zip(output_zip, project_root, required_dirs, required_files)
    print(f"\nSUCCESS: Created {output_zip}")
    print("This version has ZERO duplicate files and ZERO backslashes.")
