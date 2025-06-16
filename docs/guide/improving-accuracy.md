# Improving Accuracy

Quick guide to better transcriptions.

## Methods Overview

1. **Vocabulary Files** - Define custom terms
2. **Context Files** - Per-audio customization  
3. **Raw Content** - Extract from documents
4. **Beam Search** - Better decoding
5. **Model Selection** - Choose appropriate model

## Custom Vocabulary

### Quick Start

```bash
# Create vocab.txt with your terms
echo "PyTorch
TensorFlow  
kubernetes -> Kubernetes
ML:Machine Learning" > vocab.txt

# Use it
python main.py -i ./inputs --vocabulary-file vocab.txt
```

### Format

```text
# Corrections
java script -> JavaScript

# Acronyms
API:Application Programming Interface  

# Phrases
"machine learning pipeline"

# Terms to boost
microservices
```

## Context Files

### Per-Audio Context

Create `audio.wav.context.json`:

```json
{
  "vocabulary": ["domain", "specific", "terms"],
  "corrections": {"mistranscribed": "correct"},
  "speakers": {"SPEAKER_00": "Alice"},
  "acronyms": {"ROI": "Return on Investment"}
}
```

### Raw Content Extraction

```bash
# From single file
python main.py -i ./inputs --content-file meeting_agenda.html

# From directory  
python main.py -i ./inputs --content-dir ./docs

# Companion files (auto-detected)
meeting.wav.content.txt
presentation.wav.content.html
```

## Beam Search

| Beam Size | Speed | Accuracy | Use Case |
|-----------|-------|----------|----------|
| 1 | Fast | Good | Real-time, drafts |
| 4-8 | Medium | Better | Default, balanced |
| 16+ | Slow | Best | Final transcripts |

```bash
python main.py -i ./inputs --beam-size 16
```

## Model Selection

| Model | Best For |
|-------|----------|
| `nvidia/parakeet-tdt-0.6b-v2` | Speed, general content |
| `stt_en_conformer_ctc_small` | Quick processing |
| `stt_en_conformer_ctc_large` | Maximum accuracy |

## Domain Templates

<details>
<summary>Software Development</summary>

```text
# Common terms
GitHub
JavaScript
TypeScript
Kubernetes
microservices

# Corrections  
git hub -> GitHub
java script -> JavaScript

# Acronyms
API:Application Programming Interface
CLI:Command Line Interface
```

</details>

<details>
<summary>AI/ML</summary>

```text
# Terms
PyTorch
TensorFlow
embeddings
transformer

# Corrections
pie torch -> PyTorch

# Acronyms
LLM:Large Language Model
RAG:Retrieval Augmented Generation
```

</details>

## Tips

1. **Start simple** - Add only terms you hear mistranscribed
2. **Use context** - Meeting agendas, slides, notes
3. **Test iteratively** - Process small samples first
4. **Combine methods** - Vocabulary + context + beam search

## Quick Wins

```bash
# For technical content
python main.py -i ./inputs \
  --vocabulary-file tech_terms.txt \
  --beam-size 8

# For meetings with agenda
python main.py -i ./inputs \
  --content-file agenda.html \
  --create-context-templates
```