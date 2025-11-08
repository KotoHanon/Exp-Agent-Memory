"""
Command-line script to convert TEX or idea JSON files to standardized ResearchInput format.

Usage:
    python convert_script.py input.tex --output output.json
    python convert_script.py idea.json --output output.txt --format text
"""

import argparse
import sys
from pathlib import Path

from src.agents.experiment_agent.input_processing.converters import (
    load_research_input,
    save_research_input_as_json,
    save_research_input_as_text,
)


def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert TEX or idea JSON files to standardized ResearchInput format"
    )

    parser.add_argument(
        "input_file", type=str, help="Path to input file (.tex or .json)"
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to output file (default: input_file with new extension)",
    )

    parser.add_argument(
        "--format",
        "-f",
        type=str,
        choices=["json", "text"],
        default="json",
        help="Output format: json or text (default: json)",
    )

    parser.add_argument(
        "--encoding",
        "-e",
        type=str,
        default="utf-8",
        help="File encoding (default: utf-8)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print verbose output"
    )

    return parser.parse_args()


def main(args):
    """Main conversion logic."""
    input_path = Path(args.input_file)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    if args.verbose:
        print(f"Loading input file: {input_path}")

    # Load and convert
    try:
        research_input = load_research_input(input_path, encoding=args.encoding)

        if args.verbose:
            print(f"Successfully loaded as input_type: {research_input.input_type}")
            if research_input.title:
                print(f"Title: {research_input.title}")
    except Exception as e:
        print(f"Error loading input file: {e}")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        if args.format == "json":
            output_path = input_path.with_suffix(".standardized.json")
        else:
            output_path = input_path.with_suffix(".standardized.txt")

    # Save output
    try:
        if args.format == "json":
            save_research_input_as_json(
                research_input, output_path, encoding=args.encoding
            )
        else:
            save_research_input_as_text(
                research_input, output_path, encoding=args.encoding
            )

        print(f"Successfully converted to: {output_path}")

        if args.verbose:
            print(f"\nInput type: {research_input.input_type}")
            print(f"Content length: {len(research_input.content)} characters")
            if research_input.metadata:
                print(f"Metadata keys: {list(research_input.metadata.keys())}")

    except Exception as e:
        print(f"Error saving output file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    args = get_args()
    main(args)
