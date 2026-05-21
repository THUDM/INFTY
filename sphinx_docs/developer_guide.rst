Developer Guide
===============

Adding an Optimizer
-------------------

1. Choose an optimizer family directory under ``src/infty/optim/``.
2. Implement the optimizer class.
3. Export it in ``src/infty/optim/__init__.py``.
4. Add tests and examples.
5. Update the user guide and API reference.

Adding a Plotting Utility
-------------------------

1. Add a function under ``src/infty/plot/``.
2. Accept an explicit ``output_dir``.
3. Return saved artifact paths.
4. Export the function in ``src/infty/plot/__init__.py``.
