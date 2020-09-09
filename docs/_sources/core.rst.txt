Engine
======

DeepConcolic's test input generation algorithm is embodied in an
*Engine*, whose role is to fulfill a coverage *Criterion* by
generating inputs using an *Analyzer*.

The main entry point for constructing engines is :func:`engine.setup`.
The base classes of criteria and analyzers are
:class:`engine.Criterion` and :class:`engine.Analyzer`, although the
latter comes in two flavors depending on whether the search for inputs
is based on a given ``root`` sample
(:class:`engine.Analyzer4RootedSearch`) or based on a pre-defined
sample set (:class:`engine.Analyzer4FreeSearch`).

.. automodule:: engine

Norms
=====

.. automodule:: norms

Utilities
=========
		   
.. automodule:: utils
