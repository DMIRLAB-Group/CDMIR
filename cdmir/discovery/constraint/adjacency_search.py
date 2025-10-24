import logging
from itertools import combinations, permutations

from cdmir.graph import PDAG


def adjacency_search(indep_test, var_names, alpha=0.05, verbose=False):
    """
        This method gradually removes unnecessary edges through conditional independence testing,
        ultimately obtaining a sparse undirected graph (based on PDAG) that reflects causal relationships between variables,
        and records the separation set of each pair of nodes

      :param indep_test: Example of Conditional Independence Test
      :param var_names: List of variable names
      :param alpha: Significance level for independence tests (default: 0.05)
      :param verbose: Whether to print progress (default: False)

       Returns:
           PDAG: Partially directed acyclic graph after edge trimming
           dict: Separation sets for node pairs
    """
    # Create undirected complete graph
    g = PDAG(var_names)
    g.create_complete_undirected_graph()

    # Initialize empty separation sets for all node pairs
    sep_set = {(node_u, node_v): set() for node_u, node_v in permutations(g.nodes, 2)}
    n = len(var_names)
    # Test for conditional independence and update the separation set
    for condition_size in range(0, n - 1):
        remove_edges = []
        for node_u, node_v in permutations(var_names, 2):
            if not g.is_connected(node_u, node_v):
                continue
            adj_u = list(g.get_neighbours(node_u))
            if node_v in adj_u:
                adj_u.remove(node_v)
            for nodes_z in combinations(adj_u, condition_size):
                pval = indep_test.test(node_u, node_v, nodes_z)[0]
                # Update Separation Set
                if pval >= alpha:
                    remove_edges.append((node_u, node_v))
                    sep_set[(node_u, node_v)] |= set(nodes_z)
                    sep_set[(node_v, node_u)] |= set(nodes_z)
        # Remove edges that failed independence tests
        for node_u, node_v in remove_edges:
            if g.is_connected(node_u, node_v):
                g.remove_edge(node_u, node_v)
                if verbose:
                    logging.info(f'Adjacency Search: Remove edge between {node_u} and {node_v}.')

    return g, sep_set


def fast_adjacency_search(independence, var_names, alpha=0.05, verbose=False):
    """
        Consistent with the adjacency search function,
        but improving efficiency through optimizing the search method of the condition set,
        it is suitable for fast adjacency search when there are many variables.

        :param indep_test: Example of Conditional Independence Test
        :param var_names: List of variable names
        :param alpha: Significance level for independence tests (default: 0.05)
        :param verbose: Whether to print progress (default: False)

       Returns:
           PDAG: Partially directed acyclic graph after edge trimming
           dict: Separation sets for node pairs
       """
    # Create undirected complete graph
    g = PDAG(var_names)
    g.create_complete_undirected_graph()
    # Initialize empty separation sets for all node pairs
    sep_set = {(node_u, node_v): set() for node_u, node_v in permutations(g.nodes, 2)}
    n = len(var_names)
    # Test for conditional independence and update the separation set
    for condition_size in range(0, n - 1):
        def get_sep_result(node_u, node_v):
            sep_result = None

            # Neighborhood node search
            def sep_search(sep_result, adj):
                for nodes_z in combinations(adj, condition_size):
                    pval = independence.test(node_u, node_v, nodes_z)[0]
                    if pval >= alpha:
                        if sep_result is None:
                            sep_result = node_u, node_v, set(nodes_z)
                        else:
                            sep_result = node_u, node_v, sep_result[2] | set(nodes_z)
                        sep_set[(node_u, node_v)] |= set(nodes_z)
                        sep_set[(node_v, node_u)] |= set(nodes_z)
                return sep_result

            sep_result = sep_search(sep_result, (node for node in g.get_neighbours(node_u) if node != node_v))
            sep_result = sep_search(sep_result, (node for node in g.get_neighbours(node_v) if node != node_u))

            return sep_result

        nodes_u, nodes_v = (edge[0] for edge in g.edges), (edge[1] for edge in g.edges)
        for result in map(get_sep_result, nodes_u, nodes_v):
            if result is None:
                continue
            # Remove edge and update separation set
            g.remove_edge(result[0], result[1])
            sep_set[result[0], result[1]] |= result[2]
    return g, sep_set
