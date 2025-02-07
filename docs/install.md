# Installation

Leaspy requires Python >= 3.9, < 3.12.

Whether you wish to install a released version of Leaspy, or to install its development version, it is **highly recommended** to use a virtual environment to install the project and its dependencies.

There exists multiple solutions for that, the most common option is to use `conda`:

```bash
conda create --name leaspy python=3.10
conda activate leaspy
```

## Install a released version

To install the latest version of Leaspy:

```bash
pip install leaspy
```

## Install in development mode

If you haven't done it already, create and activate a dedicated environment (see the beginning of the installation section). 

### Clone the repository

To install the project in development mode, you first need to get the source code by cloning the project's repository:

```bash
git clone git@gitlab.com:icm-institute/aramislab/leaspy.git
cd leaspy
```

### Install poetry

This project relies on [poetry](https://python-poetry.org) that you would need to install (see the [official instructions](https://python-poetry.org/docs/#installation)).

It is recommended install it in a dedicated environment, separated from the one in which you will install Leaspy and its dependencies. One possibility is to install it with a tool called [pipx](https://pipx.pypa.io/stable/).

If you don't have `pipx` installed, already, you can follow the [official installation guidelines](https://pipx.pypa.io/stable/installation/).

In short, you can do:

```bash
pip install pipx
pipx ensurepath
pipx install poetry
```

### Install Leaspy and its dependencies

Install leaspy in development mode:

```bash
poetry install
```

### Install the pre-commit hook

Once you have installed Leaspy in development mode, do not forget to install the [pre-commit](https://pre-commit.com) hook in order to automatically format and lint your commits:

```bash
pipx install pre-commit
pre-commit install
```

## Notebook configuration

After installation, you can run the examples [here](./nutshell.md).

To do so, in your ``leaspy`` environment, you can download ``ipykernel`` to use ``leaspy`` with ``jupyter`` notebooks:

```bash
conda install ipykernel
python -m ipykernel install --user --name=leaspy
```

Now, you can open ``jupyter lab`` or ``jupyter notebook`` and select the ``leaspy`` kernel.
