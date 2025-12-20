DAG2CPDAG
==============

Convert a DAG to its corresponding CPDAG.

Usage
--------
.. code-block:: python

    from cdmir.graph import dag2cpdag
    CPDAG = dag2cpdag(G)

Parameters
---------------------
**G**: Directed Acyclic Graph.

Returns
--------------
**CPDAG**: Completed Partially Directed Acyclic Graph.
