"""Command-line entrypoint for Phase 1 CV ingestion, OCR, and text cleaning."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src import config
from src.extractor import extract_directory, extract_text
from src.models import ExtractionError, ExtractionResult

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for CV extraction.

    Returns:
        Configured :class:`argparse.ArgumentParser` instance.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Extract and clean text from CV files (.pdf, .png, .jpg, .jpeg, .docx) "
            "or a directory of CVs using PyMuPDF, OpenCV, Tesseract 5, and python-docx."
        )
    )
    parser.add_argument(
        "target",
        type=Path,
        help="Path to a single CV file or a directory containing CV files.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Output structured ExtractionResult payload(s) as UTF-8 JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional UTF-8 file path to save the rendered output.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Traverse subdirectories recursively when target is a directory.",
    )
    parser.add_argument(
        "--threshold",
        action="store_true",
        help="Enable adaptive thresholding during OCR preprocessing (default: off).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=config.OCR_DPI,
        help=f"PDF rasterization DPI for OCR fallback (default: {config.OCR_DPI}).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging on stderr.",
    )
    return parser


def format_results_table(results: list[ExtractionResult]) -> str:
    """Format extraction results as a human-readable ASCII summary table.

    Args:
        results: List of :class:`ExtractionResult` instances.

    Returns:
        Multi-line summary table string.
    """
    if not results:
        return "No supported CV files found."

    name_width = max(len("Filename"), *(len(r.filename) for r in results))
    header = (
        f"{'Filename':<{name_width}} | {'Method':<10} | {'Pages':>5} | "
        f"{'Chars':>7} | {'Time (s)':>8} | Warnings"
    )
    divider = "-" * len(header)
    lines = [header, divider]

    for result in results:
        warning_text = "; ".join(result.warnings) if result.warnings else "-"
        lines.append(
            f"{result.filename:<{name_width}} | {result.method:<10} | "
            f"{result.page_count:>5} | {len(result.cleaned_text):>7} | "
            f"{result.processing_time_sec:>8.2f} | {warning_text}"
        )

    return "\n".join(lines)


def _configure_logging(verbose: bool) -> None:
    """Configure root logging to stderr so stdout stays clean for JSON piping."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s:%(name)s:%(message)s",
        stream=sys.stderr,
        force=True,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the CV extraction CLI.

    Args:
        argv: Optional list of command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (``0`` on success, ``1`` on handled error).
    """
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)

    target: Path = args.target
    if not target.exists():
        sys.stderr.write(f"Error: Target path not found: {target}\n")
        return 1

    try:
        if target.is_dir():
            results = extract_directory(
                target,
                recursive=args.recursive,
                ocr_dpi=args.dpi,
                apply_threshold=args.threshold,
            )
            payload: object = [item.to_dict() for item in results]
        else:
            single = extract_text(
                target,
                ocr_dpi=args.dpi,
                apply_threshold=args.threshold,
            )
            results = [single]
            payload = single.to_dict()
    except (FileNotFoundError, NotADirectoryError, ExtractionError) as exc:
        logger.debug("Extraction failed.", exc_info=True)
        sys.stderr.write(f"Error: {exc}\n")
        return 1

    rendered = (
        json.dumps(payload, ensure_ascii=False, indent=2)
        if args.as_json
        else format_results_table(results)
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")

    sys.stdout.write(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
