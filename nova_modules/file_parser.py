import csv
from pathlib import Path


class NovaFileParser:
    """
    Parse uploaded files into text to provide context for NovaMind.
    Supported types: .txt, .pdf, .docx, .csv, .tsv, .py
    """

    def parse(self, file_path: str) -> str:
        """
        Main entry point for parsing any supported file type.

        Args:
            file_path: Absolute or relative path to the file.

        Returns:
            Extracted text content or an error message if unsupported/failed.
        """
        path = Path(file_path)
        if not path.exists():
            return f"Error: File not found at {file_path}"

        suffix = path.suffix.lower()

        try:
            if suffix == ".txt":
                return path.read_text(encoding="utf-8", errors="ignore")

            elif suffix == ".pdf":
                return self._parse_pdf(path)

            elif suffix == ".docx":
                return self._parse_docx(path)

            elif suffix in (".csv", ".tsv"):
                return self._parse_csv(path)

            elif suffix == ".py":
                return f"```python\n{path.read_text()}\n```"

            else:
                return f"Unsupported file type: {suffix}"

        except Exception as e:
            return f"Error parsing {suffix} file: {e!s}"

    def _parse_pdf(self, path: Path) -> str:
        """Extract text from a PDF file using PyMuPDF (fitz)."""
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(str(path))
            text = "\n".join(page.get_text() for page in doc)
            doc.close()
            return text
        except ImportError:
            return "[PDF parsing requires: pip install PyMuPDF]"
        except Exception as e:
            return f"[Error parsing PDF: {e!s}]"

    def _parse_docx(self, path: Path) -> str:
        """Extract text from a DOCX file using python-docx."""
        try:
            from docx import Document

            doc = Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs)
        except ImportError:
            return "[DOCX parsing requires: pip install python-docx]"
        except Exception as e:
            return f"[Error parsing DOCX: {e!s}]"

    def _parse_csv(self, path: Path) -> str:
        """Extract text from a CSV/TSV file (limit to first 100 lines)."""
        try:
            delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
            rows = []
            with open(path, encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f, delimiter=delimiter)
                for i, row in enumerate(reader):
                    if i >= 100:
                        break
                    rows.append(", ".join(row))
            return "\n".join(rows)
        except Exception as e:
            return f"[Error parsing CSV/TSV: {e!s}]"
