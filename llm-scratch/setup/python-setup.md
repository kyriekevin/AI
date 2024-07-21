# Python Setup Tips

There are serval different ways to set up a Python environment. Here, I am illustrating my personal preference.

I am using macbook pro running macOS, but this workflow is similar for other platforms.

## 1. Install Homebrew

Homebrew is a package manager for macOS. It allows you to install and manage software packages on your machine.

To install Homebrew, open a terminal and run the following command:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

## 2. Install MiniConda

MiniConda is a lightweight version of Anaconda, a popular Python distribution. It is designed to be easy to install and use.

To install MiniConda, run the following command in your terminal:

```bash
brew install --cask miniconda
```

## 3. Create a Conda Environment

Conda is a package manager for Python that allows you to create and manage virtual environments. A virtual environment is a self-contained directory tree that contains a Python installation for a specific project.

To create a new Conda environment, run the following command in your terminal:

```bash
conda create -n LLMs python=3.10
```

> Many scientific computing libraries do not immediately support the newest version of Python. Therefore, when installing PyTorch, it's advisable to use a version of Python that is one or two releases older. For instance, if the latest version of Python is 3.13, using Python 3.10 or 3.11 is recommended.

Next, activate the environment by running the following command:

```bash
conda activate LLMs
```

## 4. Install Required Packages

Now that you have created a Conda environment and activated it, you can install the required packages.

To install the required packages, run the following command in your terminal:

```bash
conda install package-name
```

You can also still use pip to install packages. By default, pip should install packages into the active environment.
