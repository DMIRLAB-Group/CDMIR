import numpy as np
from causaldmir.utils.independence import Dsep
from causaldmir.graph.digraph import DiGraph
from causaldmir.graph.edge import Edge
from causaldmir.utils.independence import FisherZ
from causaldmir.discovery.constraint.pc import PC

def test_desp():
    np.random.seed(10)
    X = np.random.randn(300, 1) # 生成 300*1 的随机矩阵
    X_prime = np.random.randn(300, 1) #生成和X不一样的300*1的随机数矩阵
    Y = X + 0.5 * np.random.randn(300, 1)
    Z = Y + 0.5 * np.random.randn(300, 1)
    data =np.hstack((X,X_prime,Y,Z)) #将数组按水平方向堆叠起来

    cg=PC()
    cg.fit(data,indep_cls=FisherZ)
    print(cg.causal_graph)
    # print(type(cg.causal_graph))
    node_list=list(range(data.shape[1]))

    dag=DiGraph(range(len(node_list)))
    edges=[(0, 2),(2, 3)]
    for edge in edges:
        dag.add_edge(Edge(*edge)) #(*)以元组的形式传入参数

    d = Dsep(data=data,true_graph=dag)

    # 0 && 2, 2 && 3 dependent
    a,b = d.cal_stats(0,2,[3])
    print(a,b)
    if a==0.0 and b==0.0:
        print("a and b dependent")
    elif a==1.0 and b==1.0:
        print("a and b independent")
    #assert a==0.0 and b==0.0

    a, b = d.cal_stats(0, 3, [1,2])
    print(a,b)
    if a==1.0 and b==1.0 :
        print("a and b independent")
    #assert a==1.0 and b==1.0

test_desp()