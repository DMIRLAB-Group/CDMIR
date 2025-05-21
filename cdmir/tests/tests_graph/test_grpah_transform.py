import logging

from copy import deepcopy
from unittest import TestCase

from cdmir.graph import DiGraph, Edge, Graph, Mark, PDAG
from cdmir.graph.dag2cpdag import dag2cpdag
from cdmir.graph.pdag2dag import pdag2dag

logging.basicConfig(level=logging.DEBUG,
                    format=' %(levelname)s :: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

def txt2graph(filename: str) -> Graph:
    g = Graph()
    node_map = {}
    with open(filename, "r") as file:
        next_nodes_line = False
        for line in file.readlines():
            line = line.strip()
            words = line.split()
            if len(words) > 1 and words[1] == 'Nodes:':
                next_nodes_line = True
            elif len(line) > 0 and next_nodes_line:
                next_nodes_line = False
                nodes = line.split(';')
                # print(nodes)
                for node in nodes:
                    g.add_node(node)
            elif len(words) > 0 and words[0][-1] == '.':
                next_nodes_line = False
                node1 = words[1]
                node2 = words[3]
                end1 = words[2][0]
                end2 = words[2][-1]
                g.add_edge(Edge(node1, node2, to_endpotin(end1), to_endpotin(end2)))
    return g

def to_endpotin(s: str) -> Mark:
    if s == 'o':
        return Mark.Circle
    elif s == '>':
        return Mark.Arrow
    elif s == '-':
        return Mark.Tail
    else :
        raise NotImplementedError


def graph_compare(G1, G2) -> bool:
    if G1.node_set != G2.node_set: return False
    edge_set_1 = set(G1.edges)
    edge_set_2 = set(G2.edges)
    if edge_set_1 != edge_set_2 : return False
    return True

class Test_graph_transform(TestCase):

    def test_dag2cpdag(self):
        ct = 5
        for i in range(1, ct+1):
            g = txt2graph(f'cdmir/tests/testdata/dag.{i}.txt')
            dag = DiGraph(g.node_list)
            dag.add_edges([e if e.mark_u==Mark.Tail else Edge(e.node_v, e.node_u) for e in g.edges])
            cpdag = dag2cpdag(dag)
            truth_cpdag = txt2graph(f'cdmir/tests/testdata/cpdag.{i}.txt')
            assert graph_compare(cpdag, truth_cpdag)

    def test_pdag2dag(self):
        ct = 32
        for i in range(1, ct+1):
            g = txt2graph(f'cdmir/tests/testdata/graph_data/pdag.{i}.txt')
            pdag = PDAG(g.node_list)
            pdag.add_edges(g.edges)
            dag = pdag2dag(pdag)
            truth_dag = txt2graph(f'cdmir/tests/testdata/graph_data/dag.{i}.txt')
            assert graph_compare(dag, truth_dag)

