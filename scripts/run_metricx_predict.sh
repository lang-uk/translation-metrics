#!/bin/bash

# MetricX Prediction Runner
# This script runs MetricX prediction on converted JSONL files

set -e

# Default values
VERSION="24"
MODEL_SIZE="xl"
QE_MODE=false
BATCH_SIZE=1
INPUT_DIR="metricx_data"
OUTPUT_DIR="metricx_results"

# Help message
show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Run MetricX prediction on converted JSONL files.

Options:
    -v, --version VERSION       MetricX version (23 or 24, default: 24)
    -s, --size SIZE             Model size (xxl, xl, large, default: xl)
    -q, --qe                    Use QE (reference-free) mode
    -b, --batch-size SIZE       Batch size (default: 1)
    -i, --input-dir DIR         Input directory with JSONL files (default: metricx_data)
    -o, --output-dir DIR        Output directory for results (default: metricx_results)
    -f, --file FILE             Process specific file instead of directory
    --bf16                      Use bfloat16 model (MetricX-24 only)
    -h, --help                  Show this help message

Examples:
    # Run MetricX-24 XL reference-based on all files
    ./run_metricx_predict.sh -v 24 -s xl -i metricx_data

    # Run MetricX-24 XXL QE mode on specific file
    ./run_metricx_predict.sh -v 24 -s xxl -q -f metricx_data/system_metricx24_qe.jsonl

    # Run MetricX-23 reference-based
    ./run_metricx_predict.sh -v 23 -s xl -i metricx_data

EOF
}

# Parse arguments
SPECIFIC_FILE=""
USE_BF16=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -s|--size)
            MODEL_SIZE="$2"
            shift 2
            ;;
        -q|--qe)
            QE_MODE=true
            shift
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -i|--input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -f|--file)
            SPECIFIC_FILE="$2"
            shift 2
            ;;
        --bf16)
            USE_BF16=true
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

# Validate model size
if [[ ! "$MODEL_SIZE" =~ ^(xxl|xl|large)$ ]]; then
    echo "Error: Model size must be xxl, xl, or large"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Determine model path and tokenizer
if [ "$VERSION" = "24" ]; then
    if [ "$USE_BF16" = true ]; then
        MODEL_PATH="google/metricx-24-hybrid-${MODEL_SIZE}-v2p6-bfloat16"
    else
        MODEL_PATH="google/metricx-24-hybrid-${MODEL_SIZE}-v2p6"
    fi
    TOKENIZER="google/mt5-${MODEL_SIZE}"
    MAX_LENGTH=1536
    PREDICT_MODULE="metricx24.predict"
elif [ "$VERSION" = "23" ]; then
    if [ "$QE_MODE" = true ]; then
        MODEL_PATH="google/metricx-23-qe-${MODEL_SIZE}-v2p0"
    else
        MODEL_PATH="google/metricx-23-${MODEL_SIZE}-v2p0"
    fi
    TOKENIZER="google/mt5-${MODEL_SIZE}"
    MAX_LENGTH=1024
    PREDICT_MODULE="metricx23.predict"
fi

echo "========================================"
echo "MetricX-$VERSION Prediction"
echo "========================================"
echo "Model: $MODEL_PATH"
echo "Tokenizer: $TOKENIZER"
echo "QE Mode: $QE_MODE"
echo "Batch Size: $BATCH_SIZE"
echo "Max Length: $MAX_LENGTH"
echo "========================================"
echo ""

# Build QE flag
QE_FLAG=""
if [ "$QE_MODE" = true ]; then
    QE_FLAG="--qe"
fi

# Function to process a single file
process_file() {
    local input_file="$1"
    local filename=$(basename "$input_file" .jsonl)
    local output_file="$OUTPUT_DIR/${filename}_scores.jsonl"
    
    echo "→ Processing: $input_file"
    echo "  Output: $output_file"
    
    python -m "$PREDICT_MODULE" \
        --tokenizer "$TOKENIZER" \
        --model_name_or_path "$MODEL_PATH" \
        --max_input_length "$MAX_LENGTH" \
        --batch_size "$BATCH_SIZE" \
        $QE_FLAG \
        --input_file "$input_file" \
        --output_file "$output_file"
    
    if [ $? -eq 0 ]; then
        echo "✓ Success: $output_file"
    else
        echo "✗ Failed: $input_file"
        return 1
    fi
    echo ""
}

# Process files
if [ -n "$SPECIFIC_FILE" ]; then
    # Process specific file
    if [ ! -f "$SPECIFIC_FILE" ]; then
        echo "Error: File not found: $SPECIFIC_FILE"
        exit 1
    fi
    process_file "$SPECIFIC_FILE"
else
    # Process all matching files in directory
    if [ ! -d "$INPUT_DIR" ]; then
        echo "Error: Directory not found: $INPUT_DIR"
        exit 1
    fi
    
    # Determine pattern to match
    if [ "$QE_MODE" = true ]; then
        PATTERN="*_metricx${VERSION}_qe.jsonl"
    else
        PATTERN="*_metricx${VERSION}_ref.jsonl"
    fi
    
    # Find and process files
    FILES=("$INPUT_DIR"/$PATTERN)
    
    if [ ! -e "${FILES[0]}" ]; then
        echo "No files found matching pattern: $PATTERN in $INPUT_DIR"
        exit 1
    fi
    
    echo "Found ${#FILES[@]} file(s) to process"
    echo ""
    
    for file in "${FILES[@]}"; do
        process_file "$file"
    done
fi

echo "========================================"
echo "Prediction Complete!"
echo "========================================"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  - View results: cat $OUTPUT_DIR/*_scores.jsonl"
echo "  - Analyze with visualizer: python visualizer.py"

