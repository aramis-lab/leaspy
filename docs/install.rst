.. _install:

Installation
************

Package installation
--------------------

1. Leaspy requires Python >= 3.8

2. Create a dedicated environment (optional):

Using ``conda``::

    conda create --name leaspy python=3.9
    conda activate leaspy

Or using ``pyenv``::

    pyenv virtualenv leaspy
    pyenv local leaspy

3. Install ``leaspy`` with ``pip``::

    pip install leaspy

It will automatically install all needed dependencies.


Notebook configuration
----------------------

| After installation, you can run the examples in :ref:`nutshell` and in :ref:`the Leaspy API <api>`.
| To do so, in your ``leaspy`` environment, you can download ``ipykernel`` to use ``leaspy`` with ``jupyter`` notebooks

::

    conda install ipykernel
    python -m ipykernel install --user --name=leaspy

Now, you can open ``jupyter lab`` or ``jupyter notebook`` and select the ``leaspy`` kernel.

.. Testing
.. -------
..
.. You can check that ``leaspy`` tests pass on your environment by executing:
..
..     pip install pytest pytest-subtests
..     pytest tests/unit_tests
..     pytest tests/functional_tests
..
..
.. Development
.. -----------
