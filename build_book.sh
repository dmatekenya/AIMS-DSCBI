#!/bin/bash
# Script to build Jupyter Book with the latest compatible packages

# Update pip and install latest packages
pip install -U pip
pip install sphinx>=7.2.6 myst-parser>=2.0.0 jupyter-book>=1.0.0 sphinx-book-theme>=1.0.1 sphinx-togglebutton>=0.3.2 sphinx-external-toc>=1.0.0 sphinx-multitoc-numbering>=0.1.3

# Build the book
echo "Building Jupyter Book documentation..."
# Clean the build directory first to ensure fresh build
rm -rf docs/_build

# Build with clean option to ensure all files are rebuilt
jupyter-book build docs --all

# Add a cache-busting timestamp to prevent browser caching
TIMESTAMP=$(date +%s)
find docs/_build/html -name "*.html" -exec sed -i.bak "s/<head>/<head><meta http-equiv=\"Cache-Control\" content=\"no-cache, no-store, must-revalidate\" \/><meta http-equiv=\"Pragma\" content=\"no-cache\" \/><meta http-equiv=\"Expires\" content=\"0\" \/><script>document.cacheBuster = '$TIMESTAMP';<\/script>/" {} \;
find docs/_build/html -name "*.html.bak" -delete

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo "Build successful! Open docs/_build/html/index.html in your browser to view."
else
    echo "Build failed. Check the error messages above."
fi
