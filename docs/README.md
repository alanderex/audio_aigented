# Documentation

This directory contains the source files for the Audio Transcription Pipeline documentation, built with MkDocs Material.

## Structure

```
docs/
├── index.md                    # Home page
├── getting-started/           # Installation and setup guides
├── guide/                     # User guides for features
├── deployment/                # Deployment and optimization
├── development/               # Developer documentation
├── reference/                 # API reference (auto-generated)
├── gen_ref_pages.py          # Script to generate API docs
└── stylesheets/              # Custom CSS
```

## Building Documentation

### Local Development

```bash
# Install documentation dependencies
uv pip install -e ".[docs]"

# Serve documentation locally (with hot reload)
mkdocs serve

# Build static site
mkdocs build
```

The documentation will be available at `http://localhost:8000`.

### Deploying to GitHub Pages

The documentation is automatically deployed to the `gh-pages` branch when:
1. Pushing to the `main` branch
2. Manually triggering the workflow

To deploy manually:

```bash
# Deploy to gh-pages branch
mkdocs gh-deploy

# Or use the GitHub Actions workflow
# Go to Actions tab > Deploy Documentation > Run workflow
```

The documentation will be available at: `https://[username].github.io/audio_aigented/`

## Writing Documentation

### Adding New Pages

1. Create a new markdown file in the appropriate directory
2. Add the page to the navigation in `mkdocs.yml`
3. Follow the existing style and structure

### Markdown Extensions

MkDocs Material supports many markdown extensions:

- **Admonitions**: `!!! note "Title"`
- **Code blocks**: ` ```python `
- **Tables**: Standard markdown tables
- **Task lists**: `- [x] Completed task`
- **Tabs**: `=== "Tab 1"`

### API Documentation

API documentation is automatically generated from docstrings:

```python
def my_function(param: str) -> int:
    """Short description.
    
    Longer description with more details.
    
    Args:
        param: Parameter description
        
    Returns:
        Return value description
        
    Example:
        >>> my_function("test")
        42
    """
```

## Style Guide

1. Use clear, concise language
2. Include code examples where helpful
3. Add screenshots for UI elements
4. Keep paragraphs short
5. Use appropriate heading levels
6. Include "Next Steps" sections

## Updating Documentation

When making changes:

1. Update relevant documentation
2. Test locally with `mkdocs serve`
3. Ensure no broken links
4. Submit PR with documentation changes