import time
from unittest import TestCase

from .. import Graph


class TestMapTime(TestCase):
    def test_add_nodes_case1(self):
        print()
        g1 = Graph()
        t11 = time.time()
        g1.add_nodes(range(100000))
        t12 = time.time()
        print(t12 - t11)
        g2 = Graph()
        t21 = time.time()
        g2.add_nodes_map(range(100000))
        t22 = time.time()
        print(t22 - t21)
        g3 = Graph()
        t31 = time.time()
        g3.add_nodes_np(range(100000))
        t32 = time.time()
        print(t32 - t31)

        print()

        assert g1.node_list == g2.node_list == g3.node_list

    def test_add_nodes_case2(self):
        print()
        g2 = Graph()
        t21 = time.time()
        g2.add_nodes_map(range(10000))
        t22 = time.time()
        print(t22 - t21)
        g1 = Graph()
        t11 = time.time()
        g1.add_nodes(range(10000))
        t12 = time.time()
        print(t12 - t11)
        print()
        assert g1.node_list == g2.node_list
