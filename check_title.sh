#!/bin/bash
# Script to check for discrepancies between source markdown and built HTML

echo "Checking discrepancies between source markdown and built HTML..."

# Check if a specific file is provided
if [ "$1" ]; then
    MARKDOWN_FILE="$1"
    HTML_FILE="${MARKDOWN_FILE%.md}.html"
    HTML_FILE="docs/_build/html/${HTML_FILE#docs/}"
    
    # Check if the files exist
    if [ ! -f "$MARKDOWN_FILE" ]; then
        echo "Markdown file not found: $MARKDOWN_FILE"
        exit 1
    fi
    
    if [ ! -f "$HTML_FILE" ]; then
        echo "HTML file not found: $HTML_FILE"
        exit 1
    fi
    
    # Extract the title from markdown
    MD_TITLE=$(grep -m 1 "^# " "$MARKDOWN_FILE" | sed 's/^# //')
    echo "Title in markdown: '$MD_TITLE'"
    
    # Extract the title from HTML
    HTML_TITLE=$(grep -o "<title>.*</title>" "$HTML_FILE" | sed 's/<title>\(.*\) &#8212;.*/\1/')
    echo "Title in HTML: '$HTML_TITLE'"
    
    # Compare
    if [ "$MD_TITLE" = "$HTML_TITLE" ]; then
        echo "✅ Titles match!"
    else
        echo "❌ Titles don't match!"
    fi
else
    echo "Usage: $0 path/to/markdown_file.md"
    echo "Example: $0 docs/modules/module1_python_foundations.md"
fi
