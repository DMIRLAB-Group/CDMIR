import numpy as np
import pandas as pd

def ToBij():
    signed = 1
    if np.random.randn(1) >= 0:
        signed = -1
    return signed * np.random.uniform(.5, 2)


def SelectPdf(Num,data_type="exponential"):
    if data_type == "exp-non-gaussian":
        noise = np.random.uniform(-1, 1, size=Num) ** 5

    elif data_type == "gaussian":
        noise = np.random.normal(0, 1, size=Num)

    elif data_type == "laplace":
        noise =np.random.laplace(0, 1, size=Num)

    elif data_type == "exponential":  #exp-exponential
        noise = pow(np.random.exponential(scale=1, size=Num),2)

    elif data_type == "standard_exponential":
        noise =np.random.standard_exponential(size=Num)

    else: #uniform
        noise =np.random.uniform(-1, 1, size=Num)

    return noise



def Toa():
    return 0.2


def Case1(Num=3000):
    L1=SelectPdf(Num)
    L2=SelectPdf(Num)*Toa()+ToBij()*L1
    L3=SelectPdf(Num)*Toa()+ToBij()*L1+ToBij()*L2

    x1=SelectPdf(Num)*Toa()+ToBij()*L1
    x2=SelectPdf(Num)*Toa()+ToBij()*L1
    x3=SelectPdf(Num)*Toa()+ToBij()*L1


    x4=SelectPdf(Num)*Toa()+ToBij()*L2
    x5=SelectPdf(Num)*Toa()+ToBij()*L2
    x6=SelectPdf(Num)*Toa()+ToBij()*L2



    x7=SelectPdf(Num)*Toa()+ToBij()*L3
    x8=SelectPdf(Num)*Toa()+ToBij()*L3
    x9=SelectPdf(Num)*Toa()+ToBij()*L3



    data = pd.DataFrame(np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9]).T,columns=['x1','x2','x3','x4','x5','x6','x7','x8','x9'])

    data = (data-data.mean())/data.std()
    #data = data-data.mean()
    return data




def Case2(Num=3000):
    L1=SelectPdf(Num)
    L2=SelectPdf(Num)*Toa()+ToBij()*L1
    L3=SelectPdf(Num)*Toa()+ToBij()*L2
    L4=SelectPdf(Num)*Toa()+ToBij()*L3
    L5=SelectPdf(Num)*Toa()+ToBij()*L4


    x1=SelectPdf(Num)*Toa()+ToBij()*L1
    x2=SelectPdf(Num)*Toa()+ToBij()*L1

    x3=SelectPdf(Num)*Toa()+ToBij()*L2
    x4=SelectPdf(Num)*Toa()+ToBij()*L3
    x5=SelectPdf(Num)*Toa()+ToBij()*L4

    x6=SelectPdf(Num)*Toa()+ToBij()*L5
    x7=SelectPdf(Num)*Toa()+ToBij()*L5
    data = pd.DataFrame(np.array([x1,x2,x3,x4,x5,x6,x7]).T,columns=['x1','x2','x3','x4','x5','x6','x7'])

    data = (data-data.mean())/data.std()
    #data = data-data.mean()
    return data





def Case3(Num=3000):
    L1=SelectPdf(Num)
    L2=SelectPdf(Num)*Toa()+ToBij()*L1
    L3=SelectPdf(Num)*Toa()+ToBij()*L1
    L4=SelectPdf(Num)*Toa()+ToBij()*L1

    x1=SelectPdf(Num)*Toa()+ToBij()*L1
    x2=SelectPdf(Num)*Toa()+ToBij()*L1


    x3=SelectPdf(Num)*Toa()+ToBij()*L2
    x4=SelectPdf(Num)*Toa()+ToBij()*L2


    x5=SelectPdf(Num)*Toa()+ToBij()*L3
    x6=SelectPdf(Num)*Toa()+ToBij()*L3


    data = pd.DataFrame(np.array([x1,x2,x3,x4,x5,x6]).T,columns=['x1','x2','x3','x4','x5','x6'])

    data = (data-data.mean())/data.std()
    #data = data-data.mean()
    return data





def Case4(Num=3000):
    L1=SelectPdf(Num)
    L2=SelectPdf(Num)*Toa()+ToBij()*L1
    L3=SelectPdf(Num)*Toa()+ToBij()*L1



    L4=SelectPdf(Num)*Toa()+ToBij()*L2
    L5=SelectPdf(Num)*Toa()+ToBij()*L2+ToBij()*L1



    L6=SelectPdf(Num)*Toa()+ToBij()*L1
    L7=SelectPdf(Num)*Toa()+ToBij()*L1


    L8=SelectPdf(Num)*Toa()+ToBij()*L3+ToBij()*L1
    L9=SelectPdf(Num)*Toa()+ToBij()*L3


    x1=SelectPdf(Num)*Toa()+ToBij()*L4
    x2=SelectPdf(Num)*Toa()+ToBij()*L4

    x3=SelectPdf(Num)*Toa()+ToBij()*L5
    x4=SelectPdf(Num)*Toa()+ToBij()*L5


    x5=SelectPdf(Num)*Toa()+ToBij()*L6
    x6=SelectPdf(Num)*Toa()+ToBij()*L6

    x7=SelectPdf(Num)*Toa()+ToBij()*L7
    x8=SelectPdf(Num)*Toa()+ToBij()*L7

    x9=SelectPdf(Num)*Toa()+ToBij()*L8
    x10=SelectPdf(Num)*Toa()+ToBij()*L8

    x11=SelectPdf(Num)*Toa()+ToBij()*L9
    x12=SelectPdf(Num)*Toa()+ToBij()*L9
    x13=SelectPdf(Num)*Toa()+ToBij()*L9

    x14=SelectPdf(Num)*Toa()+ToBij()*L2
    x15=SelectPdf(Num)*Toa()+ToBij()*L3



    data = pd.DataFrame(np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15]).T,columns=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15'])

    data = (data-data.mean())/data.std()
    #data = data-data.mean()
    return data

