"""Turn an invoice file of any supported format into text an agent can read.

Design note: every format converges on a single text representation, and the
LLM extracts from that one representation. The alternative -- a bespoke
structural parser per format -- looks tidier but fails exactly where this
dataset is hardest. Invoice 1006 is a ``field,value`` CSV with the key ``item``
repeated three times; ``csv.DictReader`` silently keeps only the last one. Text
preserves everything, and the model reads it fine.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from ..models import SourceDocument

SUPPORTED_SUFFIXES = {".txt", ".json", ".csv", ".xml", ".pdf"}


class UnsupportedFormatError(ValueError):
    """Raised when a file extension has no loader."""


class DocumentLoadError(RuntimeError):
    """Raised when a file exists but cannot be read as an invoice."""


def _read_text(path: Path) -> str:
    """Read a text file, tolerating the encodings scanned documents arrive in."""
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise DocumentLoadError(f"Could not decode {path} as text")


def _load_json(path: Path) -> str:
    """Pretty-print parseable JSON; hand back the raw text when it is broken.

    Malformed JSON is still an invoice -- refusing the document would turn a
    finding into a crash.
    """
    raw = _read_text(path)
    try:
        return json.dumps(json.loads(raw), indent=2)
    except json.JSONDecodeError:
        return raw


def _load_csv(path: Path) -> str:
    """Render a CSV as an aligned table, preserving every row.

    Two layouts appear in the sample data and both survive this treatment:
    a two-column ``field,value`` sheet with duplicate keys (1006), and a
    row-per-line-item sheet with trailing summary rows (1007, 1015).
    """
    raw = _read_text(path)
    try:
        rows = list(csv.reader(io.StringIO(raw)))
    except csv.Error:
        return raw

    rows = [row for row in rows if any(cell.strip() for cell in row)]
    if not rows:
        raise DocumentLoadError(f"{path} contains no data rows")

    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    col_widths = [max(len(row[i].strip()) for row in padded) for i in range(width)]

    lines = []
    for row in padded:
        cells = [row[i].strip().ljust(col_widths[i]) for i in range(width)]
        lines.append("  ".join(cells).rstrip())

    return "\n".join(lines)


def _load_pdf(path: Path) -> str:
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise DocumentLoadError(
            "pdfplumber is required to read PDF invoices: pip install pdfplumber"
        ) from exc

    pages: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")

    text = "\n\n".join(pages).strip()
    if not text:
        raise DocumentLoadError(
            f"{path} yielded no extractable text -- it may be a scanned image "
            "requiring OCR, which this prototype does not perform."
        )
    return text


_LOADERS = {
    ".txt": _read_text,
    ".xml": _read_text,
    ".json": _load_json,
    ".csv": _load_csv,
    ".pdf": _load_pdf,
}


def load_document(path: str | Path) -> SourceDocument:
    """Load an invoice file into a `SourceDocument`.

    Raises `FileNotFoundError`, `UnsupportedFormatError`, or `DocumentLoadError`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Invoice file not found: {path}")
    if not path.is_file():
        raise DocumentLoadError(f"Not a file: {path}")

    suffix = path.suffix.lower()
    loader = _LOADERS.get(suffix)
    if loader is None:
        raise UnsupportedFormatError(
            f"Unsupported invoice format '{suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )

    text = loader(path)
    if not text.strip():
        raise DocumentLoadError(f"{path} produced empty text")

    return SourceDocument(path=str(path), file_format=suffix.lstrip("."), text=text)


def discover_invoices(directory: str | Path) -> list[Path]:
    """List every loadable invoice in a directory, sorted by name."""
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )
