#!/bin/bash
# Helper script to run audio transcription with Docker

# Set default directories
INPUT_DIR="${1:-/home/hendorf/code/audio_ai/app/data/input}"
OUTPUT_DIR="${2:-/home/hendorf/code/audio_ai/app/data/output}"
DEVICE="${3:-cpu}"

echo "🎙️  Audio Transcription Docker Runner"
echo "===================================="
echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo ""

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ Error: Input directory not found: $INPUT_DIR"
    exit 1
fi

# Check for WAV files
WAV_COUNT=$(find "$INPUT_DIR" -name "*.wav" -type f 2>/dev/null | wc -l)
if [ $WAV_COUNT -eq 0 ]; then
    echo "❌ Error: No .wav files found in $INPUT_DIR"
    exit 1
fi

echo "📁 Found $WAV_COUNT audio files"
echo ""

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run transcription with auto-confirm
echo "🚀 Starting transcription..."
echo "   (This may take several minutes for long audio files)"
echo ""

# Run docker compose with Y auto-response for confirmation prompt
echo "Y" | docker compose run --rm \
    -v "$INPUT_DIR:/data/inputs:ro" \
    -v "$OUTPUT_DIR:/data/outputs:rw" \
    audio-transcription \
    --input-dir /data/inputs \
    --output-dir /data/outputs \
    --device "$DEVICE"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Transcription completed!"
    echo "📄 Results saved to: $OUTPUT_DIR"
    
    # Show summary if it exists
    if [ -f "$OUTPUT_DIR/processing_summary.txt" ]; then
        echo ""
        echo "Summary:"
        echo "--------"
        cat "$OUTPUT_DIR/processing_summary.txt"
    fi
else
    echo ""
    echo "❌ Transcription failed. Check the logs above for errors."
    exit 1
fi