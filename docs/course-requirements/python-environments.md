# Python Environment Setup

This section covers how to set up Python environments for the course and install required packages.

## Virtual Environments

Virtual environments allow you to create isolated spaces for Python projects, ensuring that each project has its own dependencies regardless of what other projects need. This is essential for maintaining consistent development environments.

### Creating a Virtual Environment

Create a Python virtual environment to use for this project. The Python version used when this was developed was 3.12.

```bash
# Create virtual environment
python -m venv .venv

# Activate the environment
# On Windows:
# .venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Update pip
pip install -U pip

# Install requirements
pip install -r requirements.txt
```

### Alternative: Using Conda Environments

If you're using Anaconda, you can create a conda environment:

```bash
# Create conda environment
conda create -n dscbi python=3.12

# Activate the environment
conda activate dscbi

# Install requirements
pip install -r requirements.txt
```

## Major Packages

For the most part, we'll install packages as needed throughout the course. However, here's a list of core packages we'll require:

1. **Transformers**: Hugging Face's state-of-the-art natural language processing library
2. **PyTorch**: Deep learning framework
3. **HuggingFace Hub**: Client library for interacting with models on Hugging Face
4. **LangChain**: Framework for developing applications powered by language models

The full list of required packages is provided in the `requirements.txt` file in the repository.

## Environment Configuration

For applications working with external APIs and services, we need to manage sensitive information securely.

### Setup `.env` file

This file is important for keeping your API keys and other secrets. Create a file named `.env` in your project root:

```
# OpenAI
OPENAI_API_KEY="<Put your token here>"

# Hugging Face
HUGGINGFACEHUB_API_TOKEN="<Put your token here>"

# Twilio Credentials
TWILIO_ACCOUNT_SID="<Put your token here>"
TWILIO_AUTH_TOKEN="<Put your token here>"
TWILIO_NUMBER="<Put your token here>"

# PostgreSQL connection details
DB_USER="<Put your token here>"
DB_PASSWORD="<Put your token here>"
```

### Loading Environment Variables

To load these variables in your Python code, you'll use the `dotenv` package:

```python
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Access environment variables
api_key = os.getenv("OPENAI_API_KEY")
```
