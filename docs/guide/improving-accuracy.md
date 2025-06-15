# Improving Transcription Accuracy

This guide covers various techniques to improve transcription accuracy, especially for technical content and domain-specific terminology.

## Overview

The audio transcription pipeline provides several methods to improve accuracy:

1. **Custom Vocabulary** - Define domain-specific terms and corrections
2. **Contextual Biasing** - Boost recognition of important terms
3. **Advanced Decoding** - Use beam search and language models
4. **Post-Processing** - Apply corrections after transcription

## Custom Vocabulary

### Creating a Vocabulary File

Create a text file with your custom vocabulary using this format:

```text
# Corrections (case-insensitive)
ai -> AI
ml -> ML
java script -> JavaScript

# Acronym Expansions
API:Application Programming Interface
SDK:Software Development Kit
NLP:Natural Language Processing

# Contextual Phrases (preserved exactly)
"machine learning"
"continuous integration"
"neural network"

# Technical Terms (for boosting)
PyTorch
TensorFlow
Kubernetes
```

### Using Custom Vocabulary

```bash
# Use with command line
python main.py --vocabulary-file config/tech_vocabulary.txt

# Combine with other options
python main.py \
    --vocabulary-file config/tech_vocabulary.txt \
    --beam-size 8 \
    --input-dir ./technical_talks
```

## Advanced Decoding Options

### Beam Search

Beam search explores multiple hypotheses during decoding:

```bash
# Default (greedy decoding)
python main.py --beam-size 1

# Better accuracy (slower)
python main.py --beam-size 8

# Maximum accuracy (much slower)
python main.py --beam-size 16
```

### Configuration File Options

Add these to your `config.yaml`:

```yaml
transcription:
  # Enhanced options
  vocabulary_file: "config/tech_vocabulary.txt"
  beam_size: 8
  lm_weight: 0.5              # Language model weight (0.0-2.0)
  word_insertion_penalty: 0.0  # Penalty for extra words
  blank_penalty: 0.0          # CTC blank penalty
```

## Domain-Specific Vocabularies

### Software Development

```text
# Common corrections
git hub -> GitHub
java script -> JavaScript
type script -> TypeScript
py torch -> PyTorch
tensor flow -> TensorFlow

# Acronyms
API:Application Programming Interface
CLI:Command Line Interface
GUI:Graphical User Interface
SDK:Software Development Kit
IDE:Integrated Development Environment

# Technical terms
microservices
containerization
orchestration
serverless
kubernetes
docker
```

### Machine Learning

```text
# Corrections
a i -> AI
m l -> ML
l l m -> LLM
g p t -> GPT

# Acronyms
NLP:Natural Language Processing
CNN:Convolutional Neural Network
RNN:Recurrent Neural Network
GAN:Generative Adversarial Network
BERT:Bidirectional Encoder Representations from Transformers

# Phrases
"neural network"
"deep learning"
"machine learning"
"transfer learning"
"reinforcement learning"
"gradient descent"
"back propagation"
```

### Medical/Healthcare

```text
# Medical acronyms
MRI:Magnetic Resonance Imaging
CT:Computed Tomography
ECG:Electrocardiogram
EEG:Electroencephalogram
ICU:Intensive Care Unit

# Medical terms
anesthesia
cardiovascular
neurological
pharmaceutical
diagnosis
prognosis
```

## Programmatic Usage

### Using Enhanced Transcriber

```python
from src.audio_aigented.transcription.enhanced_asr import EnhancedASRTranscriber
from src.audio_aigented.transcription.vocabulary import VocabularyManager

# Create vocabulary manager
vocab_manager = VocabularyManager()

# Add corrections
vocab_manager.add_corrections({
    "python": "Python",
    "java script": "JavaScript",
})

# Add technical terms
vocab_manager.add_terms([
    "PyTorch", "TensorFlow", "NumPy"
])

# Create enhanced transcriber
config = load_config()
transcriber = EnhancedASRTranscriber(config)
transcriber.vocab_manager = vocab_manager

# Transcribe with corrections
result = transcriber.transcribe_audio_file(audio_file, audio_data)
```

### Extracting Vocabulary from Text

```python
# Extract technical terms from documentation
with open("technical_docs.txt", "r") as f:
    technical_text = f.read()

transcriber.add_vocabulary_from_text(
    technical_text,
    extract_technical=True
)
```

## Best Practices

### 1. Start with Common Corrections

Begin with the most common misrecognitions in your domain:

```text
# Common ASR mistakes
i o s -> iOS
oh s -> OS
sequel -> SQL
no sequel -> NoSQL
```

### 2. Use Contextual Phrases

For multi-word technical terms:

```text
"machine learning model"
"distributed systems"
"load balancer"
"message queue"
```

### 3. Balance Vocabulary Size

- Too few terms: Missing corrections
- Too many terms: Slower processing
- Optimal: 100-500 domain-specific terms

### 4. Test and Iterate

1. Transcribe sample audio
2. Identify common errors
3. Add corrections to vocabulary
4. Re-test with updated vocabulary

### 5. Combine Techniques

Best results come from combining:
- Custom vocabulary
- Larger beam size
- Appropriate model selection
- Post-processing corrections

## Performance Considerations

| Setting | Speed Impact | Accuracy Impact |
|---------|-------------|-----------------|
| Vocabulary (< 100 terms) | Minimal | Moderate |
| Vocabulary (> 500 terms) | Slight | Moderate |
| Beam size 4 | Moderate | Good |
| Beam size 8 | Significant | Better |
| Beam size 16 | Major | Best |

## Troubleshooting

### Vocabulary Not Applied

Check that:
1. Vocabulary file path is correct
2. File format matches examples
3. Terms are spelled correctly

### Over-Correction

If the system over-corrects:
1. Make corrections more specific
2. Use word boundaries in patterns
3. Test with sample audio

### Performance Issues

If transcription is too slow:
1. Reduce beam size
2. Limit vocabulary size
3. Use GPU acceleration

## Examples

### Command Line

```bash
# Basic usage with vocabulary
python main.py \
    --input-dir ./recordings \
    --vocabulary-file ./tech_vocab.txt

# High accuracy mode
python main.py \
    --input-dir ./recordings \
    --vocabulary-file ./tech_vocab.txt \
    --beam-size 12 \
    --device cuda

# Fast mode with corrections
python main.py \
    --input-dir ./recordings \
    --vocabulary-file ./minimal_vocab.txt \
    --beam-size 2
```

### Docker

```bash
docker compose run --rm \
    -v /path/to/audio:/data/inputs:ro \
    -v /path/to/vocab.txt:/data/vocab.txt:ro \
    audio-transcription \
    --vocabulary-file /data/vocab.txt \
    --beam-size 8
```