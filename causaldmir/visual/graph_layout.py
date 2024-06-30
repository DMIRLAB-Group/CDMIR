from typing import Dict

from numpy import cos, linspace, pi, sin

from causaldmir.graph import Graph


def circular_layout(graph: Graph, scale=0.45, sort_node=False) -> Dict:
    node_list = [str(node) for node in graph.nodes]

    n = len(node_list)

    if sort_node:
        node_list = sorted(node_list)
    angle = pi / 2 - linspace(0, 2 * pi, n, endpoint=False)

    return {node: (scale * cos(angle[i]), scale * sin(angle[i])) for i, node in enumerate(node_list)}
