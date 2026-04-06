import os
import zipfile
from pathlib import Path


def zip_personal_data(output_filename="personal_data.zip", source_dir="personal_data"):
    """
    Zips the personal_data directory for easy upload to Google Colab.
    """
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"Error: {source_dir} directory not found.")
        return

    print(f"Zipping {source_dir} into {output_filename}...")

    with zipfile.ZipFile(output_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _dirs, files in os.walk(source_path):
            for file in files:
                file_path = Path(root) / file
                # Don't zip the zipped file if it's in the same dir
                if file == output_filename:
                    continue
                # Calculate relative path for the zip file structure
                rel_path = file_path.relative_to(source_path.parent)
                zipf.write(file_path, rel_path)
                print(f"  Added: {rel_path}")

    print(f"\nSuccessfully created {output_filename}!")
    print("You can now upload this file to Google Colab's /content/ directory.")


if __name__ == "__main__":
    zip_personal_data()
