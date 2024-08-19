from copy import deepcopy

import numpy as np

from cdmir.graph import DiGraph, Edge
from cdmir.graph.mark import Mark
from cdmir.graph.pdag import PDAG


# function Gd = PDAG2DAG(G) % transform a PDAG to DAG
def pdag2dag(pdag: PDAG):
    nodes = pdag.node_list
    # first create a DAG that contains all the directed edges in PDAG
    pdag_copy = deepcopy(pdag)
    edges = pdag_copy.edges
    for edge in edges:
        if not ((edge.mark_u == Mark.Arrow and edge.mark_v == Mark.Tail) or (edge.mark_u == Mark.Tail and edge.mark_v == Mark.Arrow)):
            pdag_copy.remove_edge(edge.node_u, edge.node_v)

    pdag_p = deepcopy(pdag)
    inde = np.zeros(pdag_p.number_of_nodes(), dtype=np.dtype(int)) # index whether the ith node has been removed. 1:removed; 0: not
    while 0 in inde:
        for i in range(pdag_p.number_of_nodes()):
            if inde[i] == 0:
                sign = 0
                if len(np.intersect1d(np.where([pdag_p.is_arrow(pdag_p.node_list[i], pdag_p.node_list[i_parent]) for i_parent in range(pdag_p.number_of_nodes())])[0], np.where(inde == 0)[0])) == 0: # Xi has no out-going edges
                    sign = sign + 1
                    Nx = np.intersect1d(np.intersect1d(np.where([pdag_p.is_tail(pdag_p.node_list[i], pdag_p.node_list[i_tail]) for i_tail in range(pdag_p.number_of_nodes())])[0], np.where([pdag_p.is_tail(pdag_p.node_list[i_tail], pdag_p.node_list[i]) for i_tail in range(pdag_p.number_of_nodes())])[0]), np.where(inde == 0)[0]) # find the neighbors of Xi in P
                    Ax = np.intersect1d(np.union1d(np.where([pdag_p.is_arrow(pdag_p.node_list[i_parent], pdag_p.node_list[i]) for i_parent in range(pdag_p.number_of_nodes())])[0], np.where([pdag_p.is_arrow(pdag_p.node_list[i], pdag_p.node_list[i_parent]) for i_parent in range(pdag_p.number_of_nodes())])[0]), np.where(inde == 0)[0]) # find the adjacent of Xi in P
                    Ax = np.union1d(Ax, Nx)
                    if len(Nx) > 0:
                        if check2(pdag_p, Nx, Ax): # according to the original paper
                            sign = sign + 1
                    else:
                        sign = sign + 1
                if sign == 2:
                    # for each undirected edge Y-X in PDAG, insert a directed edge Y->X in G
                    for index in np.intersect1d(np.where([pdag_p.is_tail(pdag_p.node_list[i], pdag_p.node_list[i_tail]) for i_tail in range(pdag_p.number_of_nodes())])[0], np.where([pdag_p.is_tail(pdag_p.node_list[i_tail], pdag_p.node_list[i]) for i_tail in range(pdag_p.number_of_nodes())])[0]):
                        if not pdag_copy.is_connected(nodes[index], nodes[i]):
                            pdag_copy.add_edge(Edge(nodes[index], nodes[i], Mark.Tail, Mark.Arrow), overwrite=False)
                    inde[i] = 1

    d = DiGraph(pdag_copy.node_list)
    for edge in pdag_copy.edges:
        if edge.mark_u == Mark.Arrow and edge.mark_v == Mark.Tail:
            edge = Edge(edge.node_v, edge.node_u, edge.mark_v, edge.mark_u)
        d.add_edge(edge, overwrite=True)

    return d


def check2(G, Nx, Ax):
    s = 1
    for i in range(len(Nx)):
        j = np.delete(Ax, np.where(Ax == Nx[i])[0])
        if len(np.where([not G.is_connected(G.node_list[Nx[i]], G.node_list[jj]) for jj in j])[0]) != 0:
            s = 0
            break
    return s