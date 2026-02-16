"""Convert TMX data to MetricX-compatible JSONL format.

MetricX models require specific JSONL input format:
- Reference-based: {"source": "...", "hypothesis": "...", "reference": "..."}
- Reference-free (QE): {"source": "...", "hypothesis": "..."} for MetricX-23
                       {"source": "...", "hypothesis": "...", "reference": ""} for MetricX-24
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from custom_translations_loader import CustomTranslationsLoader
from tmx_parser import TMXParser


def convert_to_metricx_format(
    sources: List[str],
    hypotheses: List[str],
    references: List[str] = None,
    system_name: str = "system",
    metricx_version: int = 24,
    qe_mode: bool = False,
) -> List[Dict]:
    """
    Convert translation data to MetricX JSONL format.

    Args:
        sources: List of source sentences
        hypotheses: List of hypothesis translations
        references: List of reference translations (optional for QE)
        system_name: Name of the translation system
        metricx_version: MetricX version (23 or 24)
        qe_mode: Whether to use QE (reference-free) mode

    Returns:
        List of dictionaries ready for JSONL serialization
    """
    data = []

    for i, (src, hyp) in enumerate(zip(sources, hypotheses)):
        # Skip empty segments
        # if not src.strip() or not hyp.strip() or hyp.strip() == '---':
        #     continue

        entry = {"source": src, "hypothesis": hyp}

        # Handle references based on version and mode
        if metricx_version == 24:
            # MetricX-24 always expects reference field
            if qe_mode:
                entry["reference"] = ""
            elif references:
                ref = references[i] if i < len(references) else ""
                # if ref.strip() and ref.strip() != '---':
                entry["reference"] = ref
                # else:
                #     continue  # Skip if no valid reference in reference-based mode
            else:
                entry["reference"] = ""
        else:  # MetricX-23
            # MetricX-23 QE doesn't use reference field
            if not qe_mode and references:
                ref = references[i] if i < len(references) else ""
                # if ref.strip() and ref.strip() != '---':
                entry["reference"] = ref
                # else:
                #     continue  # Skip if no valid reference in reference-based mode

        data.append(entry)

    return data


def load_custom_translations_json(
    json_file: str, source_field: str = "text", translation_field: str = "translation"
) -> tuple[List[str], List[str]]:
    """
    Load custom translations from JSON format.

    Args:
        json_file: Path to JSON file with custom translations
        source_field: Field name for source text (default: "text")
        translation_field: Field name for translation (default: "translation")

    Returns:
        Tuple of (sources, translations) lists
    """
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    sources = []
    translations = []

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                source = (
                    item.get(source_field) or item.get("source") or item.get("text")
                )
                translation = item.get(translation_field) or item.get("translation")

                if source is not None and translation is not None:
                    sources.append(str(source))
                    translations.append(str(translation))

    return sources, translations


def convert_custom_translations_to_metricx(
    json_file: str,
    system_name: str = "custom_system",
    metricx_version: int = 24,
    qe_mode: bool = False,
    reference_file: Optional[str] = None,
    source_field: str = "text",
    translation_field: str = "translation",
) -> List[Dict]:
    """
    Convert custom translations from JSON to MetricX format.

    Args:
        json_file: Path to JSON file with custom translations
        system_name: Name for the translation system
        metricx_version: MetricX version (23 or 24)
        qe_mode: Whether to use QE (reference-free) mode
        reference_file: Optional JSON file with reference translations
        source_field: Field name for source text
        translation_field: Field name for translation

    Returns:
        List of MetricX-formatted entries
    """
    # Load hypothesis translations
    sources, hypotheses = load_custom_translations_json(
        json_file, source_field, translation_field
    )

    # Load reference translations if provided
    references = None
    if reference_file and not qe_mode:
        _, references = load_custom_translations_json(
            reference_file, source_field, translation_field
        )

    # Convert to MetricX format
    return convert_to_metricx_format(
        sources=sources,
        hypotheses=hypotheses,
        references=references,
        system_name=system_name,
        metricx_version=metricx_version,
        qe_mode=qe_mode,
    )


def save_jsonl(data: List[Dict], output_path: str):
    """Save data as JSONL file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✓ Saved {len(data)} entries to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert TMX data or custom JSON translations to MetricX-compatible JSONL format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

TMX Files:
  # Convert TMX for MetricX-24 reference-based
  python metricx_converter.py file.tmx --system UK_2020_Bohdana_Nosenok --version 24
  
  # Convert TMX for MetricX-24 QE (reference-free)
  python metricx_converter.py file.tmx --system UK_2020_Bohdana_Nosenok --version 24 --qe
  
  # Convert all TMX systems for MetricX-24
  python metricx_converter.py file.tmx --all-systems --version 24 --output-dir metricx_data
  
  # Round-robin: each TMX system evaluated against all others as references
  python metricx_converter.py file.tmx --all-systems --version 24 --round-robin --output-dir metricx_data

Custom JSON Files:
  # Convert custom JSON translations for MetricX-24 QE
  python metricx_converter.py translations.json --custom-json --system my_model --version 24 --qe

  # Convert custom JSON with references for MetricX-24 reference-based
  python metricx_converter.py hypotheses.json --custom-json --system my_model --version 24 --reference-json references.json

  # Custom field names
  python metricx_converter.py data.json --custom-json --source-field source --translation-field target --system my_model --version 24 --qe

  # Bidirectional round-robin: evaluate custom JSON vs all TMX systems in both directions
  python metricx_converter.py my_translations.json --custom-round-robin --reference-tmx file.tmx --system my_model --version 24
        """,
    )

    parser.add_argument(
        "input_file", help="Path to TMX file or JSON file with custom translations"
    )

    parser.add_argument(
        "--system",
        type=str,
        help="Target language system to convert (for TMX files) or system name (for custom JSON)",
    )

    parser.add_argument(
        "--all-systems", action="store_true", help="Convert all target language systems"
    )

    parser.add_argument(
        "--version",
        type=int,
        choices=[23, 24],
        default=24,
        help="MetricX version (default: 24)",
    )

    parser.add_argument(
        "--qe", action="store_true", help="Use QE (reference-free) mode"
    )

    parser.add_argument(
        "--reference",
        type=str,
        help="Reference system to use (for reference-based mode). If not specified with --round-robin, uses first available system.",
    )

    parser.add_argument(
        "--round-robin",
        action="store_true",
        help="Create round-robin evaluation files (each system evaluated against all others as references)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="metricx_data",
        help="Output directory for JSONL files (default: metricx_data)",
    )

    parser.add_argument(
        "--list-systems",
        action="store_true",
        help="List available target systems and exit",
    )

    # Custom JSON conversion options
    parser.add_argument(
        "--custom-json",
        action="store_true",
        help="Treat input file as custom JSON translations instead of TMX",
    )

    parser.add_argument(
        "--source-field",
        type=str,
        default="text",
        help="Field name for source text in custom JSON (default: text)",
    )

    parser.add_argument(
        "--translation-field",
        type=str,
        default="translation",
        help="Field name for translation in custom JSON (default: translation)",
    )

    parser.add_argument(
        "--reference-json",
        type=str,
        help="Path to JSON file with reference translations (for reference-based mode)",
    )

    # Round-robin options for custom JSON
    parser.add_argument(
        "--custom-round-robin",
        action="store_true",
        help="Create bidirectional round-robin evaluation between custom JSON and TMX systems",
    )

    parser.add_argument(
        "--reference-tmx",
        type=str,
        help="TMX file containing reference systems for custom JSON round-robin evaluation",
    )

    args = parser.parse_args()

    # Handle custom round-robin mode
    if args.custom_round_robin:
        if not args.reference_tmx:
            print("✗ Error: --custom-round-robin requires --reference-tmx")
            return

        print(f"\n→ Processing custom JSON round-robin")
        print(f"  Hypothesis: {args.input_file}")
        print(f"  Reference TMX: {args.reference_tmx}")

        if not args.system:
            args.system = Path(args.input_file).stem

        # Load custom hypothesis translations
        try:
            sources, hypotheses = load_custom_translations_json(
                args.input_file, args.source_field, args.translation_field
            )
            print(f"✓ Loaded {len(hypotheses)} hypothesis translations")
        except Exception as e:
            print(f"✗ Error loading hypothesis JSON: {e}")
            return

        # Load reference TMX
        try:
            tmx_parser = TMXParser(args.reference_tmx)
            tmx_parser.parse()
            ref_sources = tmx_parser.get_source_segments()
            target_langs = tmx_parser.get_target_languages()

            # Validate that sources match
            if len(sources) != len(ref_sources):
                print(
                    f"✗ Error: Custom JSON has {len(sources)} segments, TMX has {len(ref_sources)} segments"
                )
                print(
                    f"  Use --newlines-strategy in TMX parsing or adjust custom JSON to match"
                )
                return

        except Exception as e:
            print(f"✗ Error loading reference TMX: {e}")
            return

        # Create bidirectional round-robin evaluation files
        output_dir = Path(args.output_dir)
        conversion_count = 0

        print(f"\n→ Creating bidirectional round-robin evaluation files")
        print(
            f"  1. Evaluating '{args.system}' against {len(target_langs)} TMX systems as references"
        )
        print(
            f"  2. Evaluating {len(target_langs)} TMX systems against '{args.system}' as reference"
        )

        # Direction 1: Custom JSON as hypothesis vs TMX systems as references
        for ref_system in target_langs:
            ref_translations = tmx_parser.get_target_segments(ref_system)

            # Convert to MetricX format
            metricx_data = convert_to_metricx_format(
                sources=sources,
                hypotheses=hypotheses,
                references=ref_translations,
                system_name=f"{args.system}_vs_{ref_system}",
                metricx_version=args.version,
                qe_mode=False,  # Round-robin is always reference-based
            )

            # Save output
            output_file = (
                output_dir
                / f"{args.system}_vs_{ref_system}_metricx{args.version}_ref.jsonl"
            )
            save_jsonl(metricx_data, str(output_file))
            conversion_count += 1

        # Direction 2: TMX systems as hypotheses vs Custom JSON as reference
        for hyp_system in target_langs:
            hyp_translations = tmx_parser.get_target_segments(hyp_system)

            # Convert to MetricX format
            metricx_data = convert_to_metricx_format(
                sources=sources,  # Same sources
                hypotheses=hyp_translations,  # TMX system as hypothesis
                references=hypotheses,  # Custom JSON as reference
                system_name=f"{hyp_system}_vs_{args.system}",
                metricx_version=args.version,
                qe_mode=False,  # Round-robin is always reference-based
            )

            # Save output
            output_file = (
                output_dir
                / f"{hyp_system}_vs_{args.system}_metricx{args.version}_ref.jsonl"
            )
            save_jsonl(metricx_data, str(output_file))
            conversion_count += 1

        print(
            f"\n✓ Created {conversion_count} bidirectional round-robin evaluation files"
        )
        print(f"  Files saved to: {output_dir}")
        print(f"  Patterns:")
        print(f"    {args.system}_vs_{{tmx_system}}_metricx{args.version}_ref.jsonl")
        print(f"    {{tmx_system}}_vs_{args.system}_metricx{args.version}_ref.jsonl")

        print(f"\nNext steps:")
        print(f"  1. Run predictions on all round-robin files:")
        print(
            f"     ./run_metricx_predict.sh --version {args.version} --size xl --input-dir {output_dir}"
        )
        print(f"\n  2. Analyze results:")
        print(f"     python analyze_metricx_roundrobin.py --input-dir metricx_results")

        return

    # Handle custom JSON vs TMX files
    if args.custom_json:
        print(f"\n→ Processing custom JSON file: {args.input_file}")

        if not args.system:
            # Use filename as system name
            args.system = Path(args.input_file).stem

        # Load custom translations
        try:
            sources, hypotheses = load_custom_translations_json(
                args.input_file, args.source_field, args.translation_field
            )
            print(f"✓ Loaded {len(sources)} custom translations")

            # Load references if provided
            references = None
            if args.reference_json and not args.qe:
                _, references = load_custom_translations_json(
                    args.reference_json, args.source_field, args.translation_field
                )
                print(f"✓ Loaded {len(references)} reference translations")

            # Convert to MetricX format
            metricx_data = convert_to_metricx_format(
                sources=sources,
                hypotheses=hypotheses,
                references=references,
                system_name=args.system,
                metricx_version=args.version,
                qe_mode=args.qe,
            )

            # Save output
            output_file = (
                f"{args.system}_metricx_{args.version}{'_qe' if args.qe else ''}.jsonl"
            )
            if args.output_dir != "metricx_data":
                output_file = f"{args.output_dir}/{output_file}"

            save_jsonl(metricx_data, output_file)

        except Exception as e:
            print(f"✗ Error processing custom JSON: {e}")
            return

        return

    # Parse TMX file
    print(f"\n→ Parsing TMX file: {args.input_file}")
    tmx_parser = TMXParser(args.input_file)
    tmx_parser.parse()

    source_lang = tmx_parser.get_source_language()
    target_langs = tmx_parser.get_target_languages()
    sources = tmx_parser.get_source_segments()

    print(f"✓ Found {len(target_langs)} target systems")

    # List systems if requested
    if args.list_systems:
        print("\nAvailable target systems:")
        for lang in target_langs:
            print(f"  - {lang}")
        return

    # Determine which systems to convert
    if args.all_systems:
        systems_to_convert = target_langs
    elif args.system:
        if args.system not in target_langs:
            print(f"✗ Error: System '{args.system}' not found in TMX file")
            print(f"  Available systems: {', '.join(target_langs)}")
            return
        systems_to_convert = [args.system]
    else:
        print("✗ Error: Must specify --system or --all-systems")
        parser.print_help()
        return

    # Convert each system
    output_dir = Path(args.output_dir)

    # Handle round-robin mode (for reference-based evaluation)
    if args.round_robin:
        if args.qe:
            print(
                "⚠ Warning: --round-robin is only meaningful for reference-based mode, ignoring"
            )

        print(
            f"\n→ Converting to MetricX-{args.version} format (Round-Robin Reference-based)..."
        )
        print(
            f"  Each system will be evaluated against all other systems as references"
        )

        conversion_count = 0
        for hyp_system in systems_to_convert:
            hypotheses = tmx_parser.get_target_segments(hyp_system)

            # Evaluate against each other system as reference
            for ref_system in target_langs:
                if ref_system == hyp_system:
                    continue  # Skip self-comparison

                references = tmx_parser.get_target_segments(ref_system)

                # Convert to MetricX format
                data = convert_to_metricx_format(
                    sources=sources,
                    hypotheses=hypotheses,
                    references=references,
                    system_name=hyp_system,
                    metricx_version=args.version,
                    qe_mode=False,
                )

                if not data:
                    print(f"⚠ No valid data for {hyp_system} vs {ref_system}")
                    continue

                # Save to JSONL with both system names
                output_file = (
                    output_dir
                    / f"{hyp_system}_vs_{ref_system}_metricx{args.version}_ref.jsonl"
                )
                save_jsonl(data, output_file)
                conversion_count += 1

        print(f"\n✓ Created {conversion_count} round-robin evaluation files")
        print(
            f"  Pattern: {{hypothesis}}_vs_{{reference}}_metricx{args.version}_ref.jsonl"
        )

    # Handle standard single-reference mode
    else:
        # Get reference translations if needed
        references = None
        if not args.qe or args.version == 24:
            if args.reference:
                if args.reference not in target_langs:
                    print(f"✗ Error: Reference system '{args.reference}' not found")
                    return
                references = tmx_parser.get_target_segments(args.reference)
                print(f"→ Using reference system: {args.reference}")
            elif not args.qe:
                # For reference-based mode without specified reference, use first system
                ref_system = (
                    systems_to_convert[0]
                    if len(systems_to_convert) == 1
                    else target_langs[0]
                )
                references = tmx_parser.get_target_segments(ref_system)
                print(f"→ Using reference system: {ref_system}")

        mode_str = (
            f"metricx{args.version}_qe" if args.qe else f"metricx{args.version}_ref"
        )

        print(
            f"\n→ Converting to MetricX-{args.version} format ({'QE' if args.qe else 'Reference-based'})..."
        )

        for system in systems_to_convert:
            hypotheses = tmx_parser.get_target_segments(system)

            # Convert to MetricX format
            data = convert_to_metricx_format(
                sources=sources,
                hypotheses=hypotheses,
                references=references,
                system_name=system,
                metricx_version=args.version,
                qe_mode=args.qe,
            )

            if not data:
                print(f"⚠ No valid data for system: {system}")
                continue

            # Save to JSONL
            output_file = output_dir / f"{system}_{mode_str}.jsonl"
            save_jsonl(data, output_file)

    print(f"\n✓ Conversion complete!")
    print(f"  Output directory: {output_dir}")

    if not args.round_robin:
        print(f"\nNext steps:")
        if args.version == 24:
            print(f"  1. Run prediction:")
            print(f"     python -m metricx24.predict \\")
            print(f"       --tokenizer google/mt5-xl \\")
            print(f"       --model_name_or_path google/metricx-24-hybrid-xl-v2p6 \\")
            print(f"       --max_input_length 1536 \\")
            print(f"       --batch_size 1 \\")
            if args.qe:
                print(f"       --qe \\")
            print(
                f"       --input_file {output_dir}/{systems_to_convert[0]}_{mode_str}.jsonl \\"
            )
            print(
                f"       --output_file {output_dir}/{systems_to_convert[0]}_{mode_str}_scores.jsonl"
            )
        else:
            print(f"  1. Run prediction:")
            print(f"     python -m metricx23.predict \\")
            print(f"       --tokenizer google/mt5-xl \\")
            print(f"       --model_name_or_path google/metricx-23-xl-v2p0 \\")
            print(f"       --max_input_length 1024 \\")
            print(f"       --batch_size 1 \\")
            if args.qe:
                print(f"       --qe \\")
            print(
                f"       --input_file {output_dir}/{systems_to_convert[0]}_{mode_str}.jsonl \\"
            )
            print(
                f"       --output_file {output_dir}/{systems_to_convert[0]}_{mode_str}_scores.jsonl"
            )
    else:
        print(f"\nNext steps:")
        print(f"  1. Run predictions on all round-robin files:")
        print(
            f"     ./run_metricx_predict.sh --version {args.version} --size xl --input-dir {output_dir}"
        )
        print(f"\n  2. Analyze results:")
        print(f"     python analyze_metricx_roundrobin.py --input-dir metricx_results")


if __name__ == "__main__":
    main()
