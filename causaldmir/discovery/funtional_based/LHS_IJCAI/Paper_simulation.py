import numpy as np
import pandas as pd


def ToBij():
    signed = 1
    if np.random.randn(1) >= 0:
        signed = -1
    return signed * np.random.uniform(.5, 2)


def SelectPdf(Num,data_type="gaussian"):
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



#Generation non-Gaussian noise
def SelectPdf2(Num,data_type="exponential"):
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



#Measurement Model
def CaseI(Num=3000):
    L1=SelectPdf(Num)
    L2=SelectPdf2(Num)*Toa()+ToBij()*L1
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

#Latent Tree
def CaseII(Num=3000):
    L1=SelectPdf2(Num)
    L2=SelectPdf(Num)*Toa()+ToBij()*L1
    L3=SelectPdf(Num)*Toa()+ToBij()*L1
    L4=SelectPdf(Num)*Toa()+ToBij()*L1
    L5=SelectPdf(Num)*Toa()+ToBij()*L1

    x1=SelectPdf(Num)*Toa()+ToBij()*L2
    x2=SelectPdf(Num)*Toa()+ToBij()*L2
    x3=SelectPdf(Num)*Toa()+ToBij()*L2


    x4=SelectPdf(Num)*Toa()+ToBij()*L3
    x5=SelectPdf(Num)*Toa()+ToBij()*L3
    x6=SelectPdf(Num)*Toa()+ToBij()*L3



    x7=SelectPdf(Num)*Toa()+ToBij()*L4
    x8=SelectPdf(Num)*Toa()+ToBij()*L4
    x9=SelectPdf(Num)*Toa()+ToBij()*L4


    x10=SelectPdf(Num)*Toa()+ToBij()*L5
    x11=SelectPdf(Num)*Toa()+ToBij()*L5
    x12=SelectPdf(Num)*Toa()+ToBij()*L5


    data = pd.DataFrame(np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12]).T,columns=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12'])

    data = (data-data.mean())/data.std()
    #data = data-data.mean()
    return data


#Paper example
def CaseIII(Num = 3000):
    L1=SelectPdf(Num)

    L2=SelectPdf2(Num)*Toa()+ToBij()*L1

    L3=SelectPdf2(Num)*Toa()+ToBij()*L1

    L4=SelectPdf(Num)*Toa()+ToBij()*L2

    L5=SelectPdf(Num)*Toa()+ToBij()*L1+ToBij()*L2
    L6=SelectPdf(Num)*Toa()+ToBij()*L1
    L7=SelectPdf(Num)*Toa()+ToBij()*L1


    L8=SelectPdf(Num)*Toa()+ToBij()*L3


    x1=SelectPdf(Num)*Toa()+ToBij()*L4
    x2=SelectPdf(Num)*Toa()+ToBij()*L4
    x3=SelectPdf(Num)*Toa()+ToBij()*L4


    x4=SelectPdf(Num)*Toa()+ToBij()*L5
    x5=SelectPdf(Num)*Toa()+ToBij()*L5
    x6=SelectPdf(Num)*Toa()+ToBij()*L5



    x7=SelectPdf(Num)*Toa()+ToBij()*L6
    x8=SelectPdf(Num)*Toa()+ToBij()*L6
    x9=SelectPdf(Num)*Toa()+ToBij()*L6


    x10=SelectPdf(Num)*Toa()+ToBij()*L7
    x11=SelectPdf(Num)*Toa()+ToBij()*L7
    x12=SelectPdf(Num)*Toa()+ToBij()*L7


    x13=SelectPdf(Num)*Toa()+ToBij()*L8
    x14=SelectPdf(Num)*Toa()+ToBij()*L8
    x15=SelectPdf(Num)*Toa()+ToBij()*L8

    x16=SelectPdf(Num)*Toa()+ToBij()*L2
    x17=SelectPdf(Num)*Toa()+ToBij()*L2

    x18=SelectPdf(Num)*Toa()+ToBij()*L3
    x19=SelectPdf(Num)*Toa()+ToBij()*L3




    data = pd.DataFrame(np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19]).T,columns=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18','x19'])

    data = (data-data.mean())/data.std()
    #data = data-data.mean()
    return data



def CaseIV(Num = 3000):
    L1=SelectPdf(Num)
    L2=SelectPdf2(Num)*Toa()+ToBij()*L1
    L3=SelectPdf(Num)*Toa()+ToBij()*L1+ToBij()*L2
    L4=SelectPdf(Num)*Toa()+ToBij()*L1+ToBij()*L3



    x1=SelectPdf(Num)*Toa()+ToBij()*L1
    x2=SelectPdf(Num)*Toa()+ToBij()*L1
    x3=SelectPdf(Num)*Toa()+ToBij()*L1


    x4=SelectPdf(Num)*Toa()+ToBij()*L2
    x5=SelectPdf(Num)*Toa()+ToBij()*L2
    x6=SelectPdf(Num)*Toa()+ToBij()*L2



    x7=SelectPdf(Num)*Toa()+ToBij()*L3
    x8=SelectPdf(Num)*Toa()+ToBij()*L3
    x9=SelectPdf(Num)*Toa()+ToBij()*L3


    x10=SelectPdf(Num)*Toa()+ToBij()*L4
    x11=SelectPdf(Num)*Toa()+ToBij()*L4
    x12=SelectPdf(Num)*Toa()+ToBij()*L4


    data = pd.DataFrame(np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12]).T,columns=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12'])

    data = (data-data.mean())/data.std()
    #data = data-data.mean()
    return data


#general Latent Tree
def CaseV(Num = 3000):
    L1=SelectPdf2(Num)
    L2=SelectPdf(Num)*Toa()+ToBij()*L1
    L3=SelectPdf(Num)*Toa()+ToBij()*L1
    L4=SelectPdf(Num)*Toa()+ToBij()*L1
    L5=SelectPdf2(Num)*Toa()+ToBij()*L1

    L6=SelectPdf(Num)*Toa()+ToBij()*L5
    L7=SelectPdf(Num)*Toa()+ToBij()*L5
    L8=SelectPdf(Num)*Toa()+ToBij()*L5

    L9=SelectPdf(Num)*Toa()+ToBij()*L8
    L10=SelectPdf(Num)*Toa()+ToBij()*L8
    L11=SelectPdf(Num)*Toa()+ToBij()*L8


    x1=SelectPdf(Num)*Toa()+ToBij()*L2
    x2=SelectPdf(Num)*Toa()+ToBij()*L2
    x3=SelectPdf(Num)*Toa()+ToBij()*L2


    x4=SelectPdf(Num)*Toa()+ToBij()*L3
    x5=SelectPdf(Num)*Toa()+ToBij()*L3
    x6=SelectPdf(Num)*Toa()+ToBij()*L3


    x7=SelectPdf(Num)*Toa()+ToBij()*L4
    x8=SelectPdf(Num)*Toa()+ToBij()*L4
    x9=SelectPdf(Num)*Toa()+ToBij()*L4

    x10=SelectPdf(Num)*Toa()+ToBij()*L6
    x11=SelectPdf(Num)*Toa()+ToBij()*L6
    x12=SelectPdf(Num)*Toa()+ToBij()*L6

    x13=SelectPdf(Num)*Toa()+ToBij()*L7
    x14=SelectPdf(Num)*Toa()+ToBij()*L7
    x15=SelectPdf(Num)*Toa()+ToBij()*L7


    x16=SelectPdf(Num)*Toa()+ToBij()*L9
    x17=SelectPdf(Num)*Toa()+ToBij()*L9
    x18=SelectPdf(Num)*Toa()+ToBij()*L9


    x19=SelectPdf(Num)*Toa()+ToBij()*L10
    x20=SelectPdf(Num)*Toa()+ToBij()*L10
    x21=SelectPdf(Num)*Toa()+ToBij()*L10

    x22=SelectPdf(Num)*Toa()+ToBij()*L11
    x23=SelectPdf(Num)*Toa()+ToBij()*L11
    x24=SelectPdf(Num)*Toa()+ToBij()*L11


    data = pd.DataFrame(np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24]).T,columns=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18','x19','x20','x21','x22','x23','x24'])

    data = (data-data.mean())/data.std()
    #data = data-data.mean()
    return data



def CaseVI(Num = 3000):
    L1=SelectPdf2(Num)
    L2=SelectPdf(Num)*Toa()+ToBij()*L1
    L3=SelectPdf(Num)*Toa()+ToBij()*L1+ToBij()*L2
    L4=SelectPdf(Num)*Toa()+ToBij()*L1
    L5=SelectPdf(Num)*Toa()+ToBij()*L1

    x1=SelectPdf(Num)*Toa()+ToBij()*L2
    x2=SelectPdf(Num)*Toa()+ToBij()*L2
    x3=SelectPdf(Num)*Toa()+ToBij()*L2


    x4=SelectPdf(Num)*Toa()+ToBij()*L3
    x5=SelectPdf(Num)*Toa()+ToBij()*L3
    x6=SelectPdf(Num)*Toa()+ToBij()*L3



    x7=SelectPdf(Num)*Toa()+ToBij()*L4
    x8=SelectPdf(Num)*Toa()+ToBij()*L4
    x9=SelectPdf(Num)*Toa()+ToBij()*L4


    x10=SelectPdf(Num)*Toa()+ToBij()*L5
    x11=SelectPdf(Num)*Toa()+ToBij()*L5
    x12=SelectPdf(Num)*Toa()+ToBij()*L5

    x13=SelectPdf(Num)*Toa()+ToBij()*L1


    data = pd.DataFrame(np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13]).T,columns=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13'])

    data = (data-data.mean())/data.std()
    return data




def main():
    pass

if __name__ == '__main__':
    main()
