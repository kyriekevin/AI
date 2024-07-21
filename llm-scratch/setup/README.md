# Setup

The doc lists different approaches for setting up machine and using the code in the repo.

## Quickstart

If you already have a working Python environment, you can run the following commands to install the required dependencies:

```bash
pip install -r requirements.txt
```

## Setting up for Python

If you don't have a Python environment set up, you can refer the following guide to set one up.

- [python-setup](./python-setup.md)
- [python-libraries](./python-libraries.md)

## Using Lightning Studio

For a smooth development experience in the cloud, I recommend the Lightning AI Studio platform, which allows users to set up a persistent environment and use both VSCode and Jupyter Lab on cloud CPUs and GPUs.

Once you start a new Studio, you can open the terminal and execute the following setup steps to clone the repository and install the dependencies:

```bash
cd llm-scratch
pip install -r requirements.txt
```

## Using Google Colab

You can also use Google Colab to set up a Python environment and run the code in the repo. You can optionally run the code on a GPU by changing the Runtime type to GPU.
