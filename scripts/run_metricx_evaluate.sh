#!/bin/bash

# MetricX Evaluation Runner
# This script runs MetricX evaluation against WMT datasets

set -e

# Default values
VERSION="24"
DATASET="wmt24"
RESULTS_DIR="metricx_results"
OUTPUT_DIR="metricx_evaluation"

# Help message
show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Run MetricX evaluation against WMT datasets.

Options:
    -v, --version VERSION       MetricX version (23 or 24, default: 24)
    -d, --dataset DATASET       WMT dataset (wmt22, wmt23, wmt24, default: wmt24)
    -l, --lp LP                 Language pair (e.g., en-de, en-es, ja-zh)
    -i, --input DIR             Directory with prediction results (default: metricx_results)
    -o, --output DIR            Output directory for evaluation (default: metricx_evaluation)
    -f, --file FILE             Specific predictions file to evaluate
    --wmt-eval                  Run full WMT evaluation (all language pairs)
    -h, --help                  Show this help message

Examples:
    # Evaluate single language pair
    ./run_metricx_evaluate.sh -v 24 -d wmt24 -l en-de -f metricx_results/system_scores.jsonl

    # Evaluate all language pairs for WMT24
    ./run_metricx_evaluate.sh -v 24 --wmt-eval --input metricx_results

    # Evaluate MetricX-23 on WMT22
    ./run_metricx_evaluate.sh -v 23 -d wmt22 -l en-de -f metricx_results/system_scores.jsonl

Notes:
    - Predictions file must have "system_id", "segment_id", "label", "prediction" fields
    - For WMT evaluation, you need prediction files for all language pairs
    - See metricx/README.md for more details on evaluation format

EOF
}

# Parse arguments
LP=""
SPECIFIC_FILE=""
WMT_EVAL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -l|--lp)
            LP="$2"
            shift 2
            ;;
        -i|--input)
            RESULTS_DIR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -f|--file)
            SPECIFIC_FILE="$2"
            shift 2
            ;;
        --wmt-eval)
            WMT_EVAL=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate version
if [[ "$VERSION" != "23" && "$VERSION" != "24" ]]; then
    echo "Error: Version must be 23 or 24"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Determine evaluation module
if [ "$VERSION" = "24" ]; then
    EVAL_MODULE="metricx24.evaluate"
    WMT_EVAL_MODULE="metricx24.evaluate_wmt24"
else
    EVAL_MODULE="metricx23.evaluate"
    WMT_EVAL_MODULE="metricx23.evaluate_wmt23"
fi

echo "========================================"
echo "MetricX-$VERSION Evaluation"
echo "========================================"
echo "Dataset: $DATASET"
echo "Results Directory: $RESULTS_DIR"
echo "========================================"
echo ""

if [ "$WMT_EVAL" = true ]; then
    # Full WMT evaluation
    echo "→ Running full WMT$VERSION evaluation"
    echo ""
    
    # Determine required language pairs
    if [ "$VERSION" = "24" ]; then
        LP_ARGS="--en_de --en_es --ja_zh"
        REQUIRED_FILES=("en-de" "en-es" "ja-zh")
    else
        LP_ARGS="--en_de --he_en --zh_en"
        REQUIRED_FILES=("en-de" "he-en" "zh-en")
    fi
    
    # Check if all required files exist
    MISSING=false
    for lp in "${REQUIRED_FILES[@]}"; do
        # Look for files matching pattern
        file_pattern="$RESULTS_DIR/*${lp}*_scores.jsonl"
        files=($file_pattern)
        if [ ! -e "${files[0]}" ]; then
            echo "✗ Missing predictions for language pair: $lp"
            MISSING=true
        else
            echo "✓ Found predictions for $lp: ${files[0]}"
            # Use the found file
            eval "FILE_${lp//-/_}=\"${files[0]}\""
        fi
    done
    
    if [ "$MISSING" = true ]; then
        echo ""
        echo "Error: Missing prediction files for some language pairs"
        echo "Please ensure you have predictions for all required language pairs"
        exit 1
    fi
    
    echo ""
    echo "→ Running evaluation..."
    
    output_file="$OUTPUT_DIR/wmt${VERSION}_evaluation.json"
    
    if [ "$VERSION" = "24" ]; then
        python -m "$WMT_EVAL_MODULE" \
            --en_de "$FILE_en_de" \
            --en_es "$FILE_en_es" \
            --ja_zh "$FILE_ja_zh" \
            --output_file "$output_file"
    else
        python -m "$WMT_EVAL_MODULE" \
            --en_de "$FILE_en_de" \
            --he_en "$FILE_he_en" \
            --zh_en "$FILE_zh_en" \
            --output_file "$output_file"
    fi
    
    if [ $? -eq 0 ]; then
        echo "✓ Evaluation complete: $output_file"
        echo ""
        echo "Results:"
        cat "$output_file"
    else
        echo "✗ Evaluation failed"
        exit 1
    fi
    
else
    # Single language pair evaluation
    if [ -z "$LP" ]; then
        echo "Error: Language pair (-l/--lp) required for single evaluation"
        show_help
        exit 1
    fi
    
    if [ -z "$SPECIFIC_FILE" ]; then
        echo "Error: Input file (-f/--file) required for single evaluation"
        show_help
        exit 1
    fi
    
    if [ ! -f "$SPECIFIC_FILE" ]; then
        echo "Error: File not found: $SPECIFIC_FILE"
        exit 1
    fi
    
    echo "→ Evaluating language pair: $LP"
    echo "  Input: $SPECIFIC_FILE"
    
    output_file="$OUTPUT_DIR/${LP}_evaluation.json"
    
    python -m "$EVAL_MODULE" \
        --dataset "$DATASET" \
        --lp "$LP" \
        --input_file "$SPECIFIC_FILE" \
        --output_file "$output_file"
    
    if [ $? -eq 0 ]; then
        echo "✓ Evaluation complete: $output_file"
        echo ""
        echo "Results:"
        cat "$output_file"
    else
        echo "✗ Evaluation failed"
        exit 1
    fi
fi

echo ""
echo "========================================"
echo "Evaluation Complete!"
echo "========================================"
echo "Results saved to: $OUTPUT_DIR"

