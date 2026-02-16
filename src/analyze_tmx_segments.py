#!/usr/bin/env python3
"""
Script to analyze TMX file segment counts and identify discrepancies.
"""

import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


def analyze_tmx_segments(tmx_file_path: str):
    """
    Analyze TMX file and count different types of segments.

    Args:
        tmx_file_path: Path to the TMX file
    """
    print(f"Analyzing TMX file: {tmx_file_path}")
    print("=" * 60)

    # Parse TMX file
    tree = ET.parse(tmx_file_path)
    root = tree.getroot()

    # Count total translation units
    total_tus = len(root.findall(".//tu"))
    print(f"Total <tu> elements (translation units): {total_tus}")

    # Analyze each translation unit
    languages = defaultdict(int)
    empty_segments = defaultdict(int)
    placeholder_segments = defaultdict(int)
    valid_segments = defaultdict(int)

    all_segments = defaultdict(list)

    for tu in root.findall(".//tu"):
        tu_languages = []

        for tuv in tu.findall("tuv"):
            lang = tuv.get("{http://www.w3.org/XML/1998/namespace}lang")
            if lang:
                languages[lang] += 1
                tu_languages.append(lang)

                seg = tuv.find("seg")
                if seg is not None:
                    text = seg.text if seg.text else ""
                    all_segments[lang].append(text)

                    # Check if empty or placeholder
                    if not text.strip():
                        empty_segments[lang] += 1
                    elif text.strip() == "---":
                        placeholder_segments[lang] += 1
                    else:
                        valid_segments[lang] += 1

    print("\nLanguage breakdown:")
    for lang in sorted(languages.keys()):
        print(f"  {lang}: {languages[lang]} segments")

    print("\nEmpty segments (by language):")
    for lang in sorted(empty_segments.keys()):
        if empty_segments[lang] > 0:
            print(f"  {lang}: {empty_segments[lang]} empty segments")

    print('\nPlaceholder segments ("---", by language):')
    for lang in sorted(placeholder_segments.keys()):
        if placeholder_segments[lang] > 0:
            print(f"  {lang}: {placeholder_segments[lang]} placeholder segments")

    print("\nValid segments (by language):")
    for lang in sorted(valid_segments.keys()):
        print(f"  {lang}: {valid_segments[lang]} valid segments")

    # Find source language (assuming it starts with 'EN')
    source_lang = None
    for lang in languages.keys():
        if lang.startswith("EN"):
            source_lang = lang
            break

    if source_lang:
        print(f"\nSource language identified: {source_lang}")
        source_valid = valid_segments[source_lang]
        print(f"Valid source segments: {source_valid}")

        # Check if all languages have the same number of valid segments
        consistent_langs = []
        inconsistent_langs = []

        for lang in valid_segments.keys():
            if valid_segments[lang] == source_valid:
                consistent_langs.append(lang)
            else:
                inconsistent_langs.append((lang, valid_segments[lang]))

        print(f"\nLanguages with consistent segment counts ({source_valid}):")
        for lang in sorted(consistent_langs):
            print(f"  ✓ {lang}")

        if inconsistent_langs:
            print(f"\nLanguages with inconsistent segment counts:")
            for lang, count in inconsistent_langs:
                diff = count - source_valid
                print(f"  ⚠ {lang}: {count} ({'+' if diff > 0 else ''}{diff})")
    else:
        print("\nWarning: Could not identify source language (looking for 'EN' prefix)")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY:")
    print(f"Total translation units: {total_tus}")
    if source_lang:
        print(
            f"Expected valid segments per language: {valid_segments.get(source_lang, 0)}"
        )
    print(f"Total languages found: {len(languages)}")


def compare_with_txt_file(tmx_file_path: str, txt_file_path: str):
    """
    Compare TMX analysis with extracted TXT file.
    """
    print(f"\n{'=' * 60}")
    print(f"COMPARING WITH TXT FILE: {txt_file_path}")
    print(f"{'=' * 60}")

    # Count lines in TXT file
    try:
        with open(txt_file_path, "r", encoding="utf-8") as f:
            txt_lines = sum(1 for line in f if line.strip())
        print(f"Lines in TXT file: {txt_lines}")
    except FileNotFoundError:
        print(f"TXT file not found: {txt_file_path}")
        return

    # Analyze TMX
    tree = ET.parse(tmx_file_path)
    root = tree.getroot()

    # Get source language segments (similar to parse_english_tmx_to_file.py)
    source_segments = []
    for tu in root.findall(".//tu"):
        for tuv in tu.findall("tuv"):
            lang = tuv.get("{http://www.w3.org/XML/1998/namespace}lang")
            if lang and lang.startswith("EN"):  # Source language
                seg = tuv.find("seg")
                if seg is not None and seg.text and seg.text.strip():
                    text = seg.text.strip()
                    # if text != "---":  # Exclude placeholders
                    source_segments.append(text)
                break  # Found source, move to next TU

    print(
        f"Source segments extracted (non-empty, non-placeholder): {len(source_segments)}"
    )

    if txt_lines != len(source_segments):
        print(
            f"⚠ DISCREPANCY: TXT has {txt_lines} lines, TMX extraction gives {len(source_segments)} segments"
        )
        diff = txt_lines - len(source_segments)
        print(f"Difference: {'+' if diff > 0 else ''}{diff}")

        with open(txt_file_path, "r", encoding="utf-8") as f:
            lines_from_txt = [line.strip() for line in f if line.strip()]
        # run a loop to check if the segments are the same
        for i in range(len(source_segments)):
            if source_segments[i] != lines_from_txt[i]:
                print(
                    f"⚠ DISCREPANCY: TMX segment {i} is '{source_segments[i]}' but TXT segment {i} is '{lines_from_txt[i]}'"
                )
                break


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze TMX file segment counts")
    parser.add_argument("--tmx-file", help="Path to TMX file")
    parser.add_argument("--txt-file", help="Path to extracted TXT file for comparison")

    args = parser.parse_args()

    if not Path(args.tmx_file).exists():
        print(f"Error: TMX file '{args.tmx_file}' not found")
        return

    analyze_tmx_segments(args.tmx_file)

    if args.txt_file:
        if not Path(args.txt_file).exists():
            print(f"Error: TXT file '{args.txt_file}' not found")
            return
        compare_with_txt_file(args.tmx_file, args.txt_file)


if __name__ == "__main__":
    main()
