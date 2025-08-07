#!/bin/bash
# Script to perform a complete clean build of the Jupyter Book

echo "Performing a complete clean build..."

# Remove all build artifacts
rm -rf docs/_build
rm -rf docs/.jupyter_cache
find . -name "__pycache__" -type d -exec rm -rf {} +

# Update pip and install latest packages
pip install -U pip
pip install sphinx>=7.2.6 myst-parser>=2.0.0 jupyter-book>=1.0.0 sphinx-book-theme>=1.0.1 sphinx-togglebutton>=0.3.2 sphinx-external-toc>=1.0.0 sphinx-multitoc-numbering>=0.1.3

# Build with all options to ensure everything is rebuilt
echo "Building Jupyter Book documentation..."
jupyter-book build docs --all --builder html

# Add cache-busting meta tags
echo "Adding cache-busting headers..."
TIMESTAMP=$(date +%s)
find docs/_build/html -name "*.html" -exec sed -i.bak "s/<head>/<head><meta http-equiv=\"Cache-Control\" content=\"no-cache, no-store, must-revalidate\" \/><meta http-equiv=\"Pragma\" content=\"no-cache\" \/><meta http-equiv=\"Expires\" content=\"0\" \/><script>document.cacheBuster = '$TIMESTAMP';<\/script>/" {} \;
find docs/_build/html -name "*.html.bak" -delete

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo "Build successful! Open docs/_build/html/index.html in your browser to view."
else
    echo "Build failed. Check the error messages above."
fi
