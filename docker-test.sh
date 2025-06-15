#!/bin/bash
# Test script for Docker setup

echo "🔧 Testing Docker setup for Audio Transcription Pipeline"
echo "======================================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if docker compose is available
if ! docker compose version &> /dev/null; then
    echo "❌ docker compose not found. Please install Docker Compose."
    exit 1
fi

# Check NVIDIA Docker runtime
echo "🔍 Checking GPU support..."
if docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
    echo "✅ GPU support is available"
    docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️  GPU support not available. Will use CPU."
fi

# Create necessary directories
echo ""
echo "📁 Creating directories..."
mkdir -p inputs outputs models cache
echo "✅ Directories created"

# Build the Docker image
echo ""
echo "🏗️  Building Docker image..."
docker compose build

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully"
else
    echo "❌ Failed to build Docker image"
    exit 1
fi

# Test with sample audio if provided
if [ -n "$1" ]; then
    TEST_INPUT_DIR="$1"
    echo ""
    echo "🎵 Testing with audio files from: $TEST_INPUT_DIR"
    
    # Check if directory exists
    if [ ! -d "$TEST_INPUT_DIR" ]; then
        echo "❌ Directory not found: $TEST_INPUT_DIR"
        exit 1
    fi
    
    # Count WAV files
    WAV_COUNT=$(find "$TEST_INPUT_DIR" -name "*.wav" -type f 2>/dev/null | wc -l)
    echo "Found $WAV_COUNT .wav files"
    
    if [ $WAV_COUNT -gt 0 ]; then
        echo ""
        echo "🚀 Running transcription pipeline..."
        docker compose run --rm \
            -v "$TEST_INPUT_DIR:/data/inputs:ro" \
            audio-transcription
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "✅ Test completed successfully!"
            echo "📄 Check results in: ./outputs/"
            ls -la outputs/
        else
            echo "❌ Transcription failed"
            exit 1
        fi
    else
        echo "⚠️  No .wav files found in test directory"
    fi
else
    echo ""
    echo "✅ Docker setup is ready!"
    echo ""
    echo "To test with audio files, run:"
    echo "  ./docker-test.sh /path/to/audio/files"
    echo ""
    echo "Or to test with the multi-speaker audio files:"
    echo "  ./docker-test.sh /home/hendorf/code/audio_ai/app/data/input"
fi