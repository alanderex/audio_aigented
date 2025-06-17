# Improving Accuracy

This guide provides comprehensive strategies to enhance transcription accuracy using various features of the audio transcription pipeline.

## Overview

The transcription pipeline offers multiple methods to improve accuracy, each suited for different scenarios:

1. **Vocabulary Files** - Define custom terms, corrections, and domain-specific language
2. **Context Files** - Provide per-audio customization with speaker names and specific terminology
3. **Companion Content Files** - Automatically extract context from related documents
4. **Beam Search Optimization** - Tune the decoding algorithm for better results
5. **Model Selection** - Choose the appropriate model for your content type

## Custom Vocabulary Files

### Understanding Vocabulary Files

Vocabulary files help the ASR model recognize domain-specific terms, technical jargon, and proper nouns that might otherwise be mistranscribed. The system supports multiple formats within a single file:

```bash
# Create a comprehensive vocabulary file
cat > tech_vocab.txt << EOF
# Simple terms to boost recognition probability
Kubernetes
PostgreSQL
microservices
WebSocket

# Corrections for common mistranscriptions
java script -> JavaScript
pie thon -> Python
docker compose -> Docker Compose
github actions -> GitHub Actions

# Acronym expansions
API:Application Programming Interface
CI/CD:Continuous Integration/Continuous Deployment
K8s:Kubernetes
ML:Machine Learning

# Multi-word phrases (use quotes)
"machine learning pipeline"
"distributed systems architecture"
"test driven development"
EOF

# Use the vocabulary file
python main.py -i ./inputs --vocabulary-file tech_vocab.txt
```

### How Vocabulary Enhancement Works

1. **Term Boosting**: Listed terms receive higher probability scores during decoding
2. **Correction Mapping**: Mistranscribed words are automatically replaced in post-processing
3. **Acronym Handling**: Expands acronyms when spoken in full form
4. **Phrase Recognition**: Improves recognition of multi-word technical terms

### Best Practices for Vocabulary Files

- Start with terms you've noticed being mistranscribed in initial runs
- Include variations (e.g., "PostgreSQL", "postgres", "Postgres")
- Add common phrases from your domain
- Keep files organized by topic for reusability

## Context Files

### JSON Context Files

Context files provide fine-grained control over individual audio file transcriptions:

```json
{
  "vocabulary": [
    "TensorFlow",
    "PyTorch",
    "neural network",
    "backpropagation"
  ],
  "corrections": {
    "pie torch": "PyTorch",
    "tensor flow": "TensorFlow",
    "back prop": "backprop"
  },
  "speakers": {
    "SPEAKER_00": "Dr. Sarah Chen",
    "SPEAKER_01": "Prof. Michael Roberts",
    "SPEAKER_02": "John from Engineering"
  },
  "topic": "Deep Learning Architecture Discussion",
  "acronyms": {
    "CNN": "Convolutional Neural Network",
    "RNN": "Recurrent Neural Network",
    "GPU": "Graphics Processing Unit"
  },
  "phrases": [
    "gradient descent optimization",
    "batch normalization layer",
    "attention mechanism"
  ],
  "notes": "Technical discussion about model architectures, expect heavy ML terminology"
}
```

### Creating Context Templates

Generate templates for your audio files to customize later:

```bash
# Create templates for all audio files
python main.py -i ./inputs --create-context-templates

# This creates files like:
# recording1.wav.context.json
# meeting2.wav.context.json
# presentation3.wav.context.json
```

## Companion Content Files

### Automatic Context Extraction

The system automatically looks for companion files with the same base name as your audio files:

```bash
# For audio file: presentation.wav
# The system automatically finds and uses:
# - presentation.txt (notes or transcript)
# - presentation.html (slides or agenda)
# - presentation.md (markdown notes)
# - presentation.json (structured data)
```

### Multiple Directory Search

Search for companion content across multiple directories:

```bash
# Search in multiple locations
python main.py -i ./audio \
  --content-dir ./meeting-notes \
  --content-dir ./presentations \
  --content-dir ./documentation

# For audio file "quarterly-review.wav", searches:
# ./audio/quarterly-review.{txt,html,md,json}
# ./meeting-notes/quarterly-review.{txt,html,md,json}
# ./presentations/quarterly-review.{txt,html,md,json}
# ./documentation/quarterly-review.{txt,html,md,json}
```

### Content Extraction Process

The content analyzer extracts:
- **Technical terms**: Capitalized words, compound terms
- **Acronyms**: Patterns like "API" or "API (Application Programming Interface)"
- **Key phrases**: Frequently occurring multi-word combinations
- **Domain vocabulary**: Technical terms based on context

Example HTML content that would be analyzed:

```html
<h1>Q4 Engineering Review</h1>
<h2>Agenda</h2>
<ul>
  <li>Kubernetes migration status</li>
  <li>Performance improvements in PostgreSQL</li>
  <li>New microservices architecture</li>
</ul>
<p>Discussion points: CI/CD pipeline, Docker containerization, API gateway design</p>
```

## Beam Search Optimization

### Understanding Beam Search

Beam search is a decoding algorithm that explores multiple transcription hypotheses simultaneously. The beam size determines how many alternatives to consider:

| Beam Size | Processing Time | Accuracy | Memory Usage | Use Case |
|-----------|----------------|----------|--------------|----------|
| 1 | Fastest | Good | Low | Quick drafts, real-time |
| 4 | Fast | Better | Medium | Default balanced mode |
| 8 | Medium | Very Good | Medium | Important recordings |
| 16 | Slow | Excellent | High | Critical transcriptions |
| 32+ | Very Slow | Best | Very High | Research, legal records |

### Choosing the Right Beam Size

```bash
# Quick transcription for initial review
python main.py -i ./inputs --beam-size 1

# Balanced accuracy and speed (default)
python main.py -i ./inputs --beam-size 4

# High accuracy for important content
python main.py -i ./inputs --beam-size 16

# Maximum accuracy regardless of time
python main.py -i ./inputs --beam-size 32
```

### Performance Considerations

- Each doubling of beam size roughly doubles processing time
- Diminishing returns above beam size 16 for most content
- GPU memory usage increases with beam size

## Model Selection

### Available Models

| Model | Parameters | Speed | Accuracy | Best For |
|-------|------------|-------|----------|----------|
| `nvidia/parakeet-tdt-0.6b-v2` | 600M | Very Fast | Good | General content, quick processing |
| `stt_en_conformer_ctc_small` | 14M | Fastest | Moderate | Draft transcriptions |
| `stt_en_conformer_ctc_medium` | 30M | Fast | Good | Balanced performance |
| `stt_en_conformer_ctc_large` | 120M | Moderate | Excellent | Default, high accuracy |

### Model Selection Guide

```bash
# For podcasts and general speech
python main.py -i ./inputs --model-name nvidia/parakeet-tdt-0.6b-v2

# For technical presentations requiring accuracy
python main.py -i ./inputs --model-name stt_en_conformer_ctc_large

# For quick drafts or real-time processing
python main.py -i ./inputs --model-name stt_en_conformer_ctc_small
```

### Model Characteristics

- **Parakeet**: Transducer model, excellent for continuous speech
- **Conformer CTC**: Better for technical terms, structured speech
- **Large models**: More robust to accents and background noise

## Combining Methods for Maximum Accuracy

### Layered Approach

For best results, combine multiple accuracy improvement methods:

```bash
# Maximum accuracy setup
python main.py -i ./inputs \
  --vocabulary-file domain_terms.txt \
  --content-dir ./reference-materials \
  --content-dir ./meeting-notes \
  --beam-size 16 \
  --model-name stt_en_conformer_ctc_large
```

### Workflow Example

1. **Initial Run**: Process with defaults to identify issues
   ```bash
   python main.py -i ./inputs
   ```

2. **Create Context**: Generate templates and add vocabulary
   ```bash
   python main.py -i ./inputs --create-context-templates
   # Edit the .context.json files with speaker names and terms
   ```

3. **Add Domain Knowledge**: Create vocabulary file from mistranscriptions
   ```bash
   # Create vocabulary file with corrections
   echo "commonly_mistranscribed -> correct_term" > fixes.txt
   ```

4. **Final Processing**: Run with all enhancements
   ```bash
   python main.py -i ./inputs \
     --vocabulary-file fixes.txt \
     --beam-size 8
   ```

## Domain-Specific Templates

### Software Development

```text
# Software Development Vocabulary (software_vocab.txt)
# Frameworks and Libraries
React
Angular
Vue.js
Node.js
Express.js
Django
Flask
Spring Boot

# Corrections
react js -> React.js
node JS -> Node.js
java script -> JavaScript
type script -> TypeScript
git hub -> GitHub
git lab -> GitLab

# Acronyms
API:Application Programming Interface
REST:Representational State Transfer
CRUD:Create Read Update Delete
SQL:Structured Query Language
NoSQL:Not Only SQL
ORM:Object Relational Mapping
JWT:JSON Web Token
OAuth:Open Authorization

# Common Phrases
"continuous integration pipeline"
"microservices architecture"
"test driven development"
"agile methodology"
"pull request review"
"code refactoring"
```

### Data Science & Machine Learning

```text
# ML/AI Vocabulary (ml_vocab.txt)
# Frameworks
TensorFlow
PyTorch
Scikit-learn
Keras
XGBoost
Hugging Face

# Corrections
pie torch -> PyTorch
tensor flow -> TensorFlow
sci kit learn -> scikit-learn
hugging face -> Hugging Face

# Acronyms
ML:Machine Learning
DL:Deep Learning
NLP:Natural Language Processing
CV:Computer Vision
CNN:Convolutional Neural Network
RNN:Recurrent Neural Network
LSTM:Long Short Term Memory
GAN:Generative Adversarial Network
BERT:Bidirectional Encoder Representations from Transformers

# Technical Terms
"gradient descent"
"backpropagation algorithm"
"neural network architecture"
"feature engineering"
"hyperparameter tuning"
"cross validation"
"overfitting prevention"
"model deployment"
```

### Business & Finance

```text
# Business Vocabulary (business_vocab.txt)
# Terms
KPI
ROI
EBITDA
stakeholder
blockchain
cryptocurrency

# Corrections
block chain -> blockchain
crypto currency -> cryptocurrency
stake holder -> stakeholder

# Acronyms
ROI:Return on Investment
KPI:Key Performance Indicator
B2B:Business to Business
B2C:Business to Consumer
SaaS:Software as a Service
CRM:Customer Relationship Management
ERP:Enterprise Resource Planning

# Phrases
"quarterly earnings report"
"market penetration strategy"
"customer acquisition cost"
"revenue growth projection"
```

## Troubleshooting Common Issues

### Issue: Technical Terms Mistranscribed

**Symptoms**: "Kubernetes" → "cube are net ease", "PostgreSQL" → "post grey sequel"

**Solutions**:
1. Add terms to vocabulary file
2. Include both the written and phonetic versions
3. Use correction mappings

```text
# vocabulary.txt
Kubernetes
kubernetes
k8s

# Corrections
cube are net ease -> Kubernetes
cooper net ease -> Kubernetes
post grey sequel -> PostgreSQL
post gres QL -> PostgreSQL
```

### Issue: Speaker Names Not Recognized

**Symptoms**: "SPEAKER_00", "SPEAKER_01" in output instead of names

**Solutions**:
1. Create context file with speaker mappings
2. Use consistent speaker identification

```json
{
  "speakers": {
    "SPEAKER_00": "Sarah (Host)",
    "SPEAKER_01": "Dr. James Wilson",
    "SPEAKER_02": "Guest Speaker"
  }
}
```

### Issue: Acronyms Expanded Incorrectly

**Symptoms**: "API" transcribed as "a pie" or "APIs" as "a pies"

**Solutions**:
1. Add acronym definitions
2. Include plural forms
3. Add common variations

```text
# Acronym handling
API:Application Programming Interface
APIs:Application Programming Interfaces
API's:Application Programming Interface's

# Corrections
a pie -> API
a pies -> APIs
```

### Issue: Poor Quality Audio

**Symptoms**: Many [inaudible] sections, low confidence scores

**Solutions**:
1. Increase beam size for better exploration
2. Use larger model for robustness
3. Pre-process audio (noise reduction)

```bash
# For challenging audio
python main.py -i ./inputs \
  --model-name stt_en_conformer_ctc_large \
  --beam-size 32
```

## Performance vs Accuracy Trade-offs

### Quick Reference

| Priority | Configuration | Processing Time |
|----------|--------------|-----------------|
| Speed | `--model-name stt_en_conformer_ctc_small --beam-size 1` | 1x |
| Balanced | `--model-name nvidia/parakeet-tdt-0.6b-v2 --beam-size 4` | 2-3x |
| Accuracy | `--model-name stt_en_conformer_ctc_large --beam-size 8` | 4-5x |
| Maximum | `--model-name stt_en_conformer_ctc_large --beam-size 16` | 8-10x |

### Optimization Strategy

1. **Development Phase**: Use fast settings for iteration
2. **Review Phase**: Use balanced settings for proofing  
3. **Final Phase**: Use maximum accuracy for production

## Advanced Tips

### 1. Iterative Vocabulary Building

```bash
# First pass - identify issues
python main.py -i ./sample_audio

# Review output, note mistranscriptions
# Create initial vocabulary
echo "mistranscribed_word -> correct_word" > vocab_v1.txt

# Second pass with vocabulary
python main.py -i ./sample_audio --vocabulary-file vocab_v1.txt

# Refine and expand vocabulary
# Continue until satisfied
```

### 2. Context File Inheritance

Create a base context file for common settings:

```json
{
  "acronyms": {
    "AI": "Artificial Intelligence",
    "ML": "Machine Learning"
  },
  "phrases": ["machine learning", "deep learning"]
}
```

Then extend for specific files while maintaining the base settings.

### 3. Batch Processing with Different Settings

```bash
# Process different content types with appropriate settings
# Meetings with diarization
python main.py -i ./meetings --enable-diarization --beam-size 8

# Lectures without diarization for speed  
python main.py -i ./lectures --disable-diarization --beam-size 4

# Interviews with maximum accuracy
python main.py -i ./interviews --beam-size 16 --vocabulary-file interview_terms.txt
```

### 4. Quality Assurance Workflow

1. Process with standard settings
2. Review confidence scores in JSON output
3. Focus vocabulary improvements on low-confidence sections
4. Reprocess with enhanced settings
5. Compare outputs to measure improvement

## Conclusion

Improving transcription accuracy is an iterative process. Start with the default settings to establish a baseline, then progressively add vocabulary files, context information, and tune parameters based on your specific needs. The combination of domain knowledge (through vocabulary and context files) with algorithmic optimization (beam search and model selection) provides the best results for professional transcription needs.