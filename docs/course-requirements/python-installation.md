# Python Installation

In this section, we provide guidance on installing Python and setting up a suitable development environment for this course.

## Python Version

We will be using Python 3.12 for this course. Please refer to the installation options below.

### Recommended: Installation with Anaconda

[Anaconda](https://www.anaconda.com/download) is a distribution of Python that comes pre-packaged with many useful libraries for data science and machine learning. It also includes conda, a powerful package manager that makes it easy to manage environments.

- **Download Anaconda**: [https://www.anaconda.com/download](https://www.anaconda.com/download)
- For more details about Anaconda, refer to this [blog post](https://www.anaconda.com/blog)

### Alternative: Installation from Python Website

If you prefer a more minimal installation:

- **Download Python**: [https://www.python.org/downloads/](https://www.python.org/downloads/)
- Make sure to check "Add Python to PATH" during installation (on Windows)

## Python IDEs

An IDE (Integrated Development Environment) is a software application that provides programmers with tools for software development, such as a source code editor, compiler, build automation, and debugging tools.

### Jupyter Notebook and Google Colab

After installing Python, you can proceed to install Jupyter Notebook, the default IDE for data science and scientific computing. Jupyter Notebook allows you to write code and include documentation with Markdown. 

- If you installed Python via the Anaconda distribution, Jupyter Notebook comes pre-installed
- To launch Jupyter Notebook with Anaconda: Open Anaconda Navigator and click on the Jupyter Notebook icon
- To install Jupyter separately: `pip install notebook`

In addition to local Jupyter Notebook, you can also use Google Colab, which is an online Jupyter Notebook accessible via the cloud, offering free GPUs for working with LLMs and other AI-based Python programs.

### Full-Featured IDEs

While Jupyter Notebooks are excellent for interactive data science work, this course also requires building more structured applications, which benefits from a fully-featured IDE.

> 🚀 **VS Code**: Recommended IDE for this course. See [installation instructions](https://code.visualstudio.com).

When using VS Code for Python development, we recommend installing these extensions:
- Python extension by Microsoft
- Jupyter extension
- IntelliCode

**Other IDEs**
- **PyCharm**: A powerful Python IDE with comprehensive features for professional developers
- **Notepad++**: A lightweight text editor that can be used for Python (Windows only)
