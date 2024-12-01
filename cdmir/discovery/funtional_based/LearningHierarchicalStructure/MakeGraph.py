import matplotlib.pyplot as plt
import networkx as nx


def Make_graph(LatentIndex, order=[], ShareCommonChild=[]):
    clusters = []
    # g = nx.empty_graph()
    g = nx.DiGraph()

    latent_nodes = list(LatentIndex.keys())

    A = []

    for i in latent_nodes:
        g.add_node(i)
        A.append(i)
        Clu = LatentIndex[i]
        for j in Clu:
            if j not in A:
                g.add_node(j)

            g.add_edge(i, j)

    for i in range(0, len(order)):
        for j in range(i + 1, len(order)):
            if not RmoveEdgeBetweenN_factor(order[i], order[j], ShareCommonChild):
                g.add_edge(order[i], order[j])

    A = nx.nx_agraph.to_agraph(g)
    A.layout("dot")
    # A.draw("test_tetrad.png")
    pos = nx.nx_pydot.graphviz_layout(g, prog="dot")
    nx.draw_networkx(g, pos, with_labels=True, node_color='gray', node_size=400, font_size=10)
    plt.show()


# The function is to judge n-facotr relations
def RmoveEdgeBetweenN_factor(x, y, ShareCommonChild):
    t = [x, y]
    for pattern in ShareCommonChild:
        if set(t) <= set(pattern):
            return True

    return False


# this function is to transfer cluster (type=dic) and causal order (type=list) into LatentIndex type,
# which ensure the visualization of GIN's results.
# example: Cluster={'1':[['x1','x2','x3'],'2':['x4','x5','x6']}; Order=[['x1','x2','x3'],['x4','x5','x6']]
def Transfer2Index(Cluster, Order):
    # phase I: Construct the latent cluster
    key = list(Cluster.keys())
    LIndex = 1
    LatentIndex = {}

    ShareCommonChild = []

    for LatentNum in key:
        lists = Cluster[LatentNum]
        LatentNum = int(LatentNum)
        for clu in lists:
            ts = []
            for _ in range(0, LatentNum):
                Lname = 'L' + str(LIndex)
                LatentIndex[Lname] = clu
                ts.append(Lname)
                LIndex += 1
            ShareCommonChild.append(ts)

    print(ShareCommonChild)
    # phase II: Construct the causal order between latents
    LatentNames = list(LatentIndex.keys())

    RootList = []

    for root in Order:
        # find the root cluster
        for k in LatentNames:  # finding all k is equal to root
            if set(LatentIndex[k]) == set(root):  # implies that K is root
                # if k is a n-factor, it must be remove the n-latent
                RootList.append(k)

    # Transfer to complete connect direct graph

    return LatentIndex, RootList, ShareCommonChild


def tranImpureToLatentIndex(LatentIndex, Impure):
    for C in Impure:
        for i in range(0, len(C)):
            L = C[i]
            if L not in LatentIndex.keys():
                continue
            for j in range(i + 1, len(C)):
                L_j = C[j]
                Clu = LatentIndex[L]
                if L_j not in Clu:
                    Clu = set(Clu).union([L_j])
                    LatentIndex[L] = list(Clu)

    print(LatentIndex)
    return LatentIndex


def Make_graph_Impure(LatentIndex, Impure):
    LatentIndex = tranImpureToLatentIndex(LatentIndex, Impure)
    Make_graph(LatentIndex)


# GroundTruth = {'L1': ['L2', 'L5', 'L6','L7','L3'], 'L2': ['x16', 'x17', 'L4'], 'L3': ['x18', 'x19', 'L8'], 'L4': ['x10', 'x11', 'x12'], 'L5': ['x7', 'x8', 'x9'], 'L6': ['x4', 'x5', 'x6'], 'L7': ['x1', 'x2', 'x3'], 'L8': ['L5', 'L4', 'L1']}

def main():
    LatentIndex = {'L1': ['x19', 'x18', 'L3'], 'L2': ['x16', 'x17', 'L7'], 'L3': ['x13', 'x14', 'x15'],
                   'L4': ['x10', 'x11', 'x12'], 'L5': ['x7', 'x8', 'x9'], 'L6': ['x4', 'x5', 'x6'],
                   'L7': ['x1', 'x2', 'x3'], 'L8': ['L5', 'L4', 'L1']}
    Impure = [['L1', 'L3'], ['L2', 'L7'], ['L8', 'L2', 'L6']]
    LatentIndex = tranImpureToLatentIndex(LatentIndex, Impure)
    Make_graph(LatentIndex)


def main23():
    Cluster = {'1': [['x1', 'x2', 'x3']], '2': [['x4', 'x5', 'x6']]}
    Order = [['x4', 'x5', 'x6'], ['x1', 'x2', 'x3']]
    LatentIndex, RootList, ShareCommonChild = Transfer2Index(Cluster, Order)
    Make_graph(LatentIndex, RootList, ShareCommonChild)


def main2():
    LatentIndex = {'L1': ['x1', 'x2', 'x3'], 'L2': ['x4', 'x5'], 'L3': ['x6', 'x7']}
    Order = ['L1', 'L2', 'L3']
    Make_graph(LatentIndex, Order)


if __name__ == '__main__':
    main()
