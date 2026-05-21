User Guide
==========

INFTY optimizers usually wrap a standard PyTorch optimizer. The standard pattern is:

.. code-block:: python

   base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
   optimizer = C_Flat(model.parameters(), base_optimizer, model, args={"rho": 0.05})
   optimizer.set_closure(loss_fn)
   logits, loss_list = optimizer.step()

Closure Contract
----------------

A closure must return ``(logits, loss_list)``:

.. code-block:: python

   def loss_fn():
       logits = model(inputs)
       loss = criterion(logits, targets)
       return logits, [loss]

For multi-objective optimizers, return multiple scalar losses:

.. code-block:: python

   return logits, [old_loss, new_loss]

Optimizer Families
------------------

* Geometry reshaping: ``C_Flat``, ``SAM``, ``GSAM``, ``GAM``, ``LookSAM``.
* Zeroth-order updates: ``ZeroFlow``.
* Gradient filtering: ``UniGrad_FS``, ``GradVac``, ``OGD``, ``PCGrad``, ``CAGrad``.
