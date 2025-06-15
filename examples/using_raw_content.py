#!/usr/bin/env python3
"""
Example: Using raw content files to enhance transcription.

This example shows how to use meeting agendas, presentation slides,
or documentation to improve transcription accuracy.
"""

from pathlib import Path
from src.audio_aigented.pipeline import TranscriptionPipeline
from src.audio_aigented.config.manager import ConfigManager

def create_sample_meeting_agenda():
    """Create a sample meeting agenda HTML file."""
    agenda_html = """<!DOCTYPE html>
<html>
<head>
    <title>Q4 2024 AI Strategy Meeting</title>
</head>
<body>
    <h1>Quarterly AI Strategy Review</h1>
    <p>Date: December 15, 2024</p>
    <p>Attendees: Dr. Sarah Chen (CTO), Prof. Michael Brown (AI Lead), Alex Johnson (Product Manager)</p>
    
    <h2>Agenda Items</h2>
    <ol>
        <li>Review of Q3 AI Initiatives
            <ul>
                <li>LangChain integration progress</li>
                <li>RAG (Retrieval Augmented Generation) implementation</li>
                <li>Performance metrics: latency, throughput, accuracy</li>
            </ul>
        </li>
        <li>New Technology Evaluation
            <ul>
                <li>GPT-4 Turbo vs Claude 3 comparison</li>
                <li>Fine-tuning results on domain-specific data</li>
                <li>Vector database options: Pinecone, Weaviate, Qdrant</li>
            </ul>
        </li>
        <li>Q4 Roadmap
            <ul>
                <li>Kubernetes deployment for ML models</li>
                <li>MLOps pipeline with DVC and MLflow</li>
                <li>A/B testing framework for model variants</li>
            </ul>
        </li>
        <li>Budget Discussion
            <ul>
                <li>GPU cluster expansion (H100 vs A100)</li>
                <li>OpenAI API costs optimization</li>
                <li>ROI analysis for on-premise vs cloud inference</li>
            </ul>
        </li>
    </ol>
    
    <h2>Key Terms and Acronyms</h2>
    <p>LLM - Large Language Model</p>
    <p>RAG - Retrieval Augmented Generation</p>
    <p>MLOps - Machine Learning Operations</p>
    <p>DVC - Data Version Control</p>
    <p>ROI - Return on Investment</p>
    
    <h2>Action Items from Last Meeting</h2>
    <ul>
        <li>DONE: Implement semantic search with embeddings</li>
        <li>IN PROGRESS: Migrate to microservices architecture</li>
        <li>PENDING: Evaluate AutoML solutions</li>
    </ul>
</body>
</html>"""
    
    agenda_file = Path("meeting_agenda.html")
    agenda_file.write_text(agenda_html)
    print(f"Created sample agenda: {agenda_file}")
    return agenda_file

def create_technical_documentation():
    """Create a sample technical documentation file."""
    doc_content = """# AI Pipeline Technical Documentation

## Overview
Our AI pipeline leverages state-of-the-art transformer models for natural language processing tasks. The system is built on a microservices architecture deployed on Kubernetes.

## Architecture Components

### 1. Embedding Service
- Model: text-embedding-ada-002
- Vector dimension: 1536
- Similarity metric: cosine similarity
- Caching: Redis with 24h TTL

### 2. LLM Service
- Primary model: gpt-4-turbo-preview
- Fallback model: claude-3-sonnet
- Temperature: 0.7 for creative tasks, 0.2 for analytical
- Max tokens: 4096
- Streaming enabled for real-time responses

### 3. Vector Database
- Provider: Pinecone
- Index type: HNSW (Hierarchical Navigable Small World)
- Dimensions: 1536
- Metric: cosine
- Pods: 2x p2.x1 for high availability

### 4. Orchestration Layer
- Framework: LangChain
- Memory: ConversationSummaryBufferMemory
- Tools: WebSearch, Calculator, PythonREPL
- Agents: ReAct pattern with self-reflection

## Performance Metrics
- P95 latency: 250ms
- Throughput: 1000 requests/minute
- Uptime SLA: 99.9%
- Cost per 1K tokens: $0.03

## Deployment
```bash
kubectl apply -f k8s/deployment.yaml
kubectl rollout status deployment/ai-pipeline
```

## Monitoring
- Prometheus for metrics collection
- Grafana dashboards for visualization
- PagerDuty for alerting
- Datadog APM for distributed tracing

## Security Considerations
- API keys stored in HashiCorp Vault
- mTLS for inter-service communication
- Rate limiting: 100 requests per minute per user
- Input sanitization for prompt injection prevention
"""
    
    doc_file = Path("technical_docs.md")
    doc_file.write_text(doc_content)
    print(f"Created technical documentation: {doc_file}")
    return doc_file

def create_presentation_notes():
    """Create presentation speaker notes."""
    notes = """AI Strategy Presentation - Speaker Notes

Slide 1: Introduction
- Welcome everyone to our quarterly AI strategy review
- Quick round of introductions for any new attendees
- Mention that we'll be discussing both technical progress and business impact

Slide 2: Q3 Achievements
- Emphasize the 40% reduction in inference latency
- LangChain integration completed ahead of schedule
- RAG implementation showing 85% accuracy on domain-specific queries
- Highlight cost savings from prompt optimization

Slide 3: Technology Comparison
- GPT-4 Turbo: Better for complex reasoning, higher cost
- Claude 3: More consistent, better at following instructions
- Recommendation: Use both in a multi-model approach
- Show benchmark results on our test dataset

Slide 4: Vector Database Evaluation
- Pinecone: Current choice, great performance but expensive
- Weaviate: Open source option, considering for non-critical workloads
- Qdrant: Impressive benchmarks, planning POC next month

Slide 5: Q4 Roadmap
- Kubernetes migration critical for scaling
- MLOps pipeline will enable faster experimentation
- A/B testing framework essential for data-driven decisions
- Timeline: K8s by end of January, MLOps by February

Slide 6: Budget Considerations
- H100 GPUs: 2.5x performance but 3x cost of A100
- Recommend starting with A100s and upgrading based on demand
- OpenAI costs can be reduced by 30% with better caching
- On-premise becomes cost-effective at >500K requests/month

Key Discussion Points:
- Need alignment on build vs buy for vector database
- Resource allocation between research and production optimization
- Hiring plans for ML engineers in Q1 2025
- Partnership opportunities with cloud providers
"""
    
    notes_file = Path("presentation_notes.txt")
    notes_file.write_text(notes)
    print(f"Created presentation notes: {notes_file}")
    return notes_file

def main():
    """Demonstrate using raw content files for context enhancement."""
    
    # Create sample content files
    agenda_file = create_sample_meeting_agenda()
    doc_file = create_technical_documentation()
    notes_file = create_presentation_notes()
    
    # Setup configuration
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    # Enable context features
    config.transcription["enable_file_context"] = True
    
    # Initialize pipeline
    pipeline = TranscriptionPipeline(config)
    
    # Analyze content files
    content_files = [agenda_file, doc_file, notes_file]
    print(f"\nAnalyzing {len(content_files)} content files...")
    
    context = pipeline.context_manager.load_raw_content_files(content_files)
    
    # Display extracted context
    print("\n📊 Extracted Context Summary:")
    print(f"Vocabulary terms: {len(context.get('vocabulary', []))}")
    print(f"Technical terms extracted: {', '.join(context.get('vocabulary', [])[:10])}...")
    
    print(f"\nAcronyms found: {len(context.get('acronyms', {}))}")
    for acronym, expansion in list(context.get('acronyms', {}).items())[:5]:
        print(f"  - {acronym}: {expansion}")
    
    print(f"\nKey phrases: {len(context.get('phrases', []))}")
    for phrase in context.get('phrases', [])[:5]:
        print(f"  - {phrase}")
    
    # Example: Process audio with this context
    audio_file = Path("./inputs/ai_strategy_meeting.wav")
    
    if audio_file.exists():
        # Apply context and process
        if hasattr(pipeline.asr_transcriber, 'vocab_manager'):
            pipeline.context_manager.create_enhanced_vocabulary(
                pipeline.asr_transcriber.vocab_manager, context
            )
        
        print(f"\nProcessing audio with enhanced context from {len(content_files)} files...")
        result = pipeline.process_single_file(audio_file)
        
        print("\nTranscription excerpt with technical terms:")
        print(result.full_text[:500] + "...")
    else:
        print(f"\nAudio file not found: {audio_file}")
        print("Context extraction completed successfully!")
        
        # Save context for review
        import json
        with open("extracted_context.json", "w") as f:
            json.dump(context, f, indent=2)
        print("\nExtracted context saved to: extracted_context.json")
    
    # Clean up example files
    for f in content_files:
        f.unlink()
    
    print("\nExample completed!")

if __name__ == "__main__":
    main()