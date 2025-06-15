#!/bin/bash
set -e

# Function to show usage
show_usage() {
    echo "Audio Transcription Pipeline - Docker Container"
    echo ""
    echo "Usage:"
    echo "  docker-compose run audio-transcription [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --input-dir PATH     Input directory (default: /data/inputs)"
    echo "  --output-dir PATH    Output directory (default: /data/outputs)"
    echo "  --device DEVICE      Device to use: cuda or cpu (default: cuda)"
    echo "  --disable-diarization  Disable speaker diarization"
    echo "  --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Process all files in mounted input directory"
    echo "  docker-compose run audio-transcription"
    echo ""
    echo "  # Process with specific options"
    echo "  docker-compose run audio-transcription --disable-diarization"
    echo ""
    echo "  # Run interactive bash shell"
    echo "  docker-compose run audio-transcription bash"
}

# Check if help is requested
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    show_usage
    exit 0
fi

# Check if running bash or sh
if [[ "$1" == "bash" ]] || [[ "$1" == "sh" ]] || [[ "$1" == "/bin/bash" ]] || [[ "$1" == "/bin/sh" ]]; then
    exec "$@"
fi

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.free,memory.total --format=csv,noheader
    echo ""
fi

# Set default directories if not provided
if [[ "$@" != *"--input-dir"* ]]; then
    set -- "$@" --input-dir /data/inputs
fi

if [[ "$@" != *"--output-dir"* ]]; then
    set -- "$@" --output-dir /data/outputs
fi

# Create output directory if it doesn't exist
OUTPUT_DIR="/data/outputs"
for arg in "$@"; do
    if [[ $prev_arg == "--output-dir" ]]; then
        OUTPUT_DIR="$arg"
    fi
    prev_arg="$arg"
done

mkdir -p "$OUTPUT_DIR"

# Check if input directory exists and has files
INPUT_DIR="/data/inputs"
for arg in "$@"; do
    if [[ $prev_arg == "--input-dir" ]]; then
        INPUT_DIR="$arg"
    fi
    prev_arg="$arg"
done

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Error: Input directory not found: $INPUT_DIR"
    echo "Make sure to mount your input directory in docker-compose.yml"
    exit 1
fi

# Count WAV files
WAV_COUNT=$(find "$INPUT_DIR" -name "*.wav" -type f 2>/dev/null | wc -l)
if [[ $WAV_COUNT -eq 0 ]]; then
    echo "Warning: No .wav files found in $INPUT_DIR"
fi

echo "Starting Audio Transcription Pipeline..."
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Found $WAV_COUNT WAV files to process"
echo ""

# Run the main application
exec python3 main.py "$@"