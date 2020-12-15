DeepConcolic API
================

Core
----

DeepConcolic's test input generation algorithm is embodied in an
*Engine*, whose role is to fulfill a coverage *Criterion* by
generating inputs using an *Analyzer*.

.. toctree::

   core

Available Coverage Metrics & Analyzers
--------------------------------------

Several coverage metrics are currently implemented in DeepConcolic,
along with one or more associated analyzers for each of them.

.. toctree::

   nc
   mcdc
   dbnc

Internals
---------

Most analyzers above make use of various internal modules for problem
solving and encoding.

.. toctree::

   lp
   l0
