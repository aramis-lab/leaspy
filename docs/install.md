# Installation

## Package installation

1. Leaspy requires Python >= 3.8

2. Create a dedicated environment (optional):

Using ``conda``:

```bash
conda create --name leaspy python=3.9
conda activate leaspy
```

Or using ``pyenv``:

```bash
pyenv virtualenv leaspy
pyenv local leaspy
```

3. Install ``leaspy`` with ``pip``:

```bash
pip install leaspy
```

It will automatically install all needed dependencies.

## Notebook configuration

After installation, you can run the examples [here](./nutshell.md).

To do so, in your ``leaspy`` environment, you can download ``ipykernel`` to use ``leaspy`` with ``jupyter`` notebooks:

```bash
conda install ipykernel
python -m ipykernel install --user --name=leaspy
```

Now, you can open ``jupyter lab`` or ``jupyter notebook`` and select the ``leaspy`` kernel.
