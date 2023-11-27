import networkx as nx
import matplotlib.pyplot as plt

#draw graph according to LatentIndex
#LatentIndex example: LatentIndex={'L1':['x1','x2','L2'],'L2':['x3','x4']}
def Make_graph(LatentIndex):
    clusters = []
    #g = nx.empty_graph()
    g = nx.DiGraph()
    latent_nodes =list(LatentIndex.keys())
    A=[]
    for i in latent_nodes:
        g.add_node(i)
        A.append(i)
        Clu=LatentIndex[i]
        for j in Clu:
            if j not in A:
                g.add_node(j)

    for i in latent_nodes:
        Clu=LatentIndex[i]
        for j in Clu:
            g.add_edge(i, j)

    A = nx.nx_agraph.to_agraph(g)
    A.layout("dot")
    A.draw("LHM_ouput.png")
    pos = nx.nx_pydot.graphviz_layout(g, prog="dot")
    nx.draw_networkx(g, pos, with_labels=True)
    plt.show()



#test draw the graph
def test():
    LatentIndex={'L1':['x1','x2','L2'],'L2':['x3','x4']}
    Make_graph(LatentIndex)


if __name__ == '__main__':
    test()
