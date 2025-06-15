# Model Management Strategy

This document outlines the approach for managing large NVIDIA NeMo model files in this repository.

## Current Models

The pipeline uses the following pre-trained models:
- **ASR**: `stt_en_conformer_ctc_large` (~700MB)
- **VAD**: `vad_multilingual_marblenet` (~10MB)  
- **Speaker Embeddings**: `titanet-l` (~100MB)

Total size: ~810MB per model set

## Recommended Approach: Automatic Download

Models are **not** included in the repository. Instead, they are automatically downloaded on first use:

1. Models are downloaded to the `models/` directory (excluded from git)
2. The directory structure is preserved for caching
3. Subsequent runs use the cached models

### Benefits
- Keeps repository size small
- Models always up-to-date from NVIDIA's servers
- No Git LFS complexity
- Works seamlessly with Docker setup

### Setup Instructions

For users:
```bash
# Models will be auto-downloaded to models/ on first run
# Ensure you have ~1GB free space
python main.py --input-dir ./inputs

# Or with Docker (models persist in volume)
docker-compose run --rm audio-transcription
```

For developers who want to pre-download:
```python
from nemo.collections.asr.models import EncDecCTCModel
from nemo.collections.asr.models.classification_models import EncDecClassificationModel

# Pre-download models
asr_model = EncDecCTCModel.from_pretrained("stt_en_conformer_ctc_large")
vad_model = EncDecClassificationModel.from_pretrained("vad_multilingual_marblenet")
```

## Alternative: Git LFS (Not Recommended)

If you must version control models:

```bash
# Initialize Git LFS
git lfs install

# Track model files
git lfs track "models/**/*.nemo"
git add .gitattributes

# Add and commit models
git add models/
git commit -m "Add NeMo models with LFS"
```

**Drawbacks:**
- Requires Git LFS setup for all users
- Large clone sizes
- LFS bandwidth/storage costs
- Models may become outdated

## Docker Considerations

The Docker setup handles models efficiently:
- Models are stored in a persistent volume
- Shared across container runs
- Automatically downloaded if missing
- No rebuild needed when models change

```yaml
volumes:
  - ./models:/home/appuser/.cache/torch/NeMo
```

## Verification

To verify models are properly cached:

```bash
# Check model directory
ls -la models/

# Expected structure:
# models/
# ├── stt_en_conformer_ctc_large/
# │   └── <hash>/
# │       └── stt_en_conformer_ctc_large.nemo
# ├── titanet-l/
# │   └── <hash>/
# │       └── titanet-l.nemo
# └── vad_multilingual_marblenet/
#     └── <hash>/
#         └── vad_multilingual_marblenet.nemo
```

## Troubleshooting

### Models Not Downloading
- Check internet connectivity
- Verify write permissions to `models/` directory
- Ensure sufficient disk space (~1GB)
- Check NVIDIA NGC is accessible

### Using Custom Models
Edit `config/default.yaml`:
```yaml
model:
  name: "your_custom_model"  # Must be a valid NeMo model name
  
diarization:
  embedding:
    model_path: "titanet-l"  # Or path to custom embedding model
```