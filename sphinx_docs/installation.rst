Installation
============

Install from PyPI:

.. code-block:: bash

   python -m pip install infty

Install from source:

.. code-block:: bash

   git clone https://github.com/THUDM/INFTY.git
   cd INFTY
   python -m pip install -e ".[examples]"

Verify imports:

.. code-block:: bash

   python -c "from infty.optim import C_Flat, ZeroFlow, UniGrad_FS; print('INFTY import succeeded')"
