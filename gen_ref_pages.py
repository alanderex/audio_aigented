"""Generate the code reference pages and navigation."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

src_root = Path("src/audio_aigented")

for path in sorted(src_root.rglob("*.py")):
    # Skip __pycache__ and test files
    if "__pycache__" in str(path) or path.stem.startswith("test_"):
        continue
    
    # Get the module path
    module_path = path.relative_to("src").with_suffix("")
    doc_path = path.relative_to("src").with_suffix(".md")
    full_doc_path = Path("reference", doc_path)
    
    # Convert path to Python module notation
    parts = tuple(module_path.parts)
    
    # Skip __init__ files in navigation but still document them
    if parts[-1] == "__init__":
        parts = parts[:-1]
        if not parts:  # Skip the root __init__.py
            continue
    
    # Add to navigation
    nav[parts] = doc_path.as_posix()
    
    # Generate the markdown file
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"# {ident}\n\n")
        fd.write(f"::: {ident}\n")
    
    # Set the edit path
    mkdocs_gen_files.set_edit_path(full_doc_path, path)

# Write the navigation file
with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())