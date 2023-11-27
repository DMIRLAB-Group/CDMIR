#-------------------------------------------------------------------------------
# Name:        模块1
# Purpose:
#
# Author:      YY
#
# Created:     07/04/2022
# Copyright:   (c) YY 2022
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np
import pandas as pd

def ToBij2():
     ten = np.random.randint(0,2)
     s  =np.random.random()
     while abs(s) <0.5 and ten ==0:
        s  =np.random.random()
     result = ten+s
     if np.random.randint(0,10)>5:
        result = -1*result
     #print(result)
     return round(result,1)
     #return 1.2


def ToBij():
    signed = 1
    if np.random.randn(1) >= 0:
        signed = -1
    return signed * np.random.uniform(.5, 2)


def SelectPdf2(Num):
    print(' i am obserced noise!')
    noise = np.random.normal(0, 1, size=Num)
    return noise


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



def wid3(Num=3000):
    L1=SelectPdf(Num)

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


def wid4(Num=3000):
    L1=SelectPdf(Num)

    L2=SelectPdf(Num)*Toa()+ToBij()*L1
    L3=SelectPdf(Num)*Toa()+ToBij()*L1
    L4=SelectPdf(Num)*Toa()+ToBij()*L1
    L5=SelectPdf(Num)*Toa()+ToBij()*L1

    x1=SelectPdf(Num)*Toa()+ToBij()*L2
    x2=SelectPdf(Num)*Toa()+ToBij()*L2
    x3=SelectPdf(Num)*Toa()+ToBij()*L2
    x4=SelectPdf(Num)*Toa()+ToBij()*L2

    x5=SelectPdf(Num)*Toa()+ToBij()*L3
    x6=SelectPdf(Num)*Toa()+ToBij()*L3
    x7=SelectPdf(Num)*Toa()+ToBij()*L3
    x8=SelectPdf(Num)*Toa()+ToBij()*L3

    x9=SelectPdf(Num)*Toa()+ToBij()*L4
    x10=SelectPdf(Num)*Toa()+ToBij()*L4
    x11=SelectPdf(Num)*Toa()+ToBij()*L4
    x12=SelectPdf(Num)*Toa()+ToBij()*L4

    x13=SelectPdf(Num)*Toa()+ToBij()*L5
    x14=SelectPdf(Num)*Toa()+ToBij()*L5
    x15=SelectPdf(Num)*Toa()+ToBij()*L5
    x16=SelectPdf(Num)*Toa()+ToBij()*L5


    data = pd.DataFrame(np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16]).T,columns=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16'])

    data = (data-data.mean())/data.std()
    #data = data-data.mean()
    return data


def wid5(Num=3000):
    L1=SelectPdf(Num)

    L2=SelectPdf(Num)*Toa()+ToBij()*L1
    L3=SelectPdf(Num)*Toa()+ToBij()*L1
    L4=SelectPdf(Num)*Toa()+ToBij()*L1
    L5=SelectPdf(Num)*Toa()+ToBij()*L1

    x1=SelectPdf(Num)*Toa()+ToBij()*L2
    x2=SelectPdf(Num)*Toa()+ToBij()*L2
    x3=SelectPdf(Num)*Toa()+ToBij()*L2
    x4=SelectPdf(Num)*Toa()+ToBij()*L2
    x5=SelectPdf(Num)*Toa()+ToBij()*L2

    x6=SelectPdf(Num)*Toa()+ToBij()*L3
    x7=SelectPdf(Num)*Toa()+ToBij()*L3
    x8=SelectPdf(Num)*Toa()+ToBij()*L3
    x9=SelectPdf(Num)*Toa()+ToBij()*L3
    x10=SelectPdf(Num)*Toa()+ToBij()*L3

    x11=SelectPdf(Num)*Toa()+ToBij()*L4
    x12=SelectPdf(Num)*Toa()+ToBij()*L4
    x13=SelectPdf(Num)*Toa()+ToBij()*L4
    x14=SelectPdf(Num)*Toa()+ToBij()*L4
    x15=SelectPdf(Num)*Toa()+ToBij()*L4

    x16=SelectPdf(Num)*Toa()+ToBij()*L5
    x17=SelectPdf(Num)*Toa()+ToBij()*L5
    x18=SelectPdf(Num)*Toa()+ToBij()*L5
    x19=SelectPdf(Num)*Toa()+ToBij()*L5
    x20=SelectPdf(Num)*Toa()+ToBij()*L5


    data = pd.DataFrame(np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20]).T,columns=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18','x19','x20'])

    data = (data-data.mean())/data.std()
    #data = data-data.mean()
    return data


def wid6(Num=3000):
    L1=SelectPdf(Num)

    L2=SelectPdf(Num)*Toa()+ToBij()*L1
    L3=SelectPdf(Num)*Toa()+ToBij()*L1
    L4=SelectPdf(Num)*Toa()+ToBij()*L1
    L5=SelectPdf(Num)*Toa()+ToBij()*L1

    x1=SelectPdf(Num)*Toa()+ToBij()*L2
    x2=SelectPdf(Num)*Toa()+ToBij()*L2
    x3=SelectPdf(Num)*Toa()+ToBij()*L2
    x4=SelectPdf(Num)*Toa()+ToBij()*L2
    x5=SelectPdf(Num)*Toa()+ToBij()*L2
    x6=SelectPdf(Num)*Toa()+ToBij()*L2

    x7=SelectPdf(Num)*Toa()+ToBij()*L3
    x8=SelectPdf(Num)*Toa()+ToBij()*L3
    x9=SelectPdf(Num)*Toa()+ToBij()*L3
    x10=SelectPdf(Num)*Toa()+ToBij()*L3
    x11=SelectPdf(Num)*Toa()+ToBij()*L3
    x12=SelectPdf(Num)*Toa()+ToBij()*L3

    x13=SelectPdf(Num)*Toa()+ToBij()*L4
    x14=SelectPdf(Num)*Toa()+ToBij()*L4
    x15=SelectPdf(Num)*Toa()+ToBij()*L4
    x16=SelectPdf(Num)*Toa()+ToBij()*L4
    x17=SelectPdf(Num)*Toa()+ToBij()*L4
    x18=SelectPdf(Num)*Toa()+ToBij()*L4

    x19=SelectPdf(Num)*Toa()+ToBij()*L5
    x20=SelectPdf(Num)*Toa()+ToBij()*L5
    x21=SelectPdf(Num)*Toa()+ToBij()*L5
    x22=SelectPdf(Num)*Toa()+ToBij()*L5
    x23=SelectPdf(Num)*Toa()+ToBij()*L5
    x24=SelectPdf(Num)*Toa()+ToBij()*L5



    data = pd.DataFrame(np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24]).T,columns=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18','x19','x20','x21','x22','x23','x24'])

    data = (data-data.mean())/data.std()
    #data = data-data.mean()
    return data


def dep4(Num=3000):
    L1=SelectPdf(Num)

    L2=SelectPdf(Num)*Toa()+ToBij()*L1
    L3=SelectPdf(Num)*Toa()+ToBij()*L1
    L4=SelectPdf(Num)*Toa()+ToBij()*L1



    L5=SelectPdf(Num)*Toa()+ToBij()*L2
    L6=SelectPdf(Num)*Toa()+ToBij()*L2

    L7=SelectPdf(Num)*Toa()+ToBij()*L3
    L8=SelectPdf(Num)*Toa()+ToBij()*L3

    L9=SelectPdf(Num)*Toa()+ToBij()*L4
    L10=SelectPdf(Num)*Toa()+ToBij()*L4



    x1=SelectPdf(Num)*Toa()+ToBij()*L5
    x2=SelectPdf(Num)*Toa()+ToBij()*L5
    x3=SelectPdf(Num)*Toa()+ToBij()*L6
    x4=SelectPdf(Num)*Toa()+ToBij()*L6
    x5=SelectPdf(Num)*Toa()+ToBij()*L7
    x6=SelectPdf(Num)*Toa()+ToBij()*L7
    x7=SelectPdf(Num)*Toa()+ToBij()*L8
    x8=SelectPdf(Num)*Toa()+ToBij()*L8

    x9=SelectPdf(Num)*Toa()+ToBij()*L9
    x10=SelectPdf(Num)*Toa()+ToBij()*L9
    x11=SelectPdf(Num)*Toa()+ToBij()*L10
    x12=SelectPdf(Num)*Toa()+ToBij()*L10




    data = pd.DataFrame(np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12]).T,columns=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12'])

    data = (data-data.mean())/data.std()
    #data = data-data.mean()
    return data


def dep5(Num=3000):
    L1=SelectPdf(Num)

    L2=SelectPdf(Num)*Toa()+ToBij()*L1
    L3=SelectPdf(Num)*Toa()+ToBij()*L1
    L4=SelectPdf(Num)*Toa()+ToBij()*L1



    L5=SelectPdf(Num)*Toa()+ToBij()*L2
    L6=SelectPdf(Num)*Toa()+ToBij()*L2
    L7=SelectPdf(Num)*Toa()+ToBij()*L3
    L8=SelectPdf(Num)*Toa()+ToBij()*L3
    L9=SelectPdf(Num)*Toa()+ToBij()*L4
    L10=SelectPdf(Num)*Toa()+ToBij()*L4


    L11=SelectPdf(Num)*Toa()+ToBij()*L5
    L12=SelectPdf(Num)*Toa()+ToBij()*L5
    L13=SelectPdf(Num)*Toa()+ToBij()*L6
    L14=SelectPdf(Num)*Toa()+ToBij()*L6
    L15=SelectPdf(Num)*Toa()+ToBij()*L7
    L16=SelectPdf(Num)*Toa()+ToBij()*L7
    L17=SelectPdf(Num)*Toa()+ToBij()*L8
    L18=SelectPdf(Num)*Toa()+ToBij()*L8
    L19=SelectPdf(Num)*Toa()+ToBij()*L9
    L20=SelectPdf(Num)*Toa()+ToBij()*L9
    L21=SelectPdf(Num)*Toa()+ToBij()*L10
    L22=SelectPdf(Num)*Toa()+ToBij()*L10




    x1=SelectPdf(Num)*Toa()+ToBij()*L11
    x2=SelectPdf(Num)*Toa()+ToBij()*L11
    x3=SelectPdf(Num)*Toa()+ToBij()*L12
    x4=SelectPdf(Num)*Toa()+ToBij()*L12
    x5=SelectPdf(Num)*Toa()+ToBij()*L13
    x6=SelectPdf(Num)*Toa()+ToBij()*L13
    x7=SelectPdf(Num)*Toa()+ToBij()*L14
    x8=SelectPdf(Num)*Toa()+ToBij()*L14
    x9=SelectPdf(Num)*Toa()+ToBij()*L15
    x10=SelectPdf(Num)*Toa()+ToBij()*L15
    x11=SelectPdf(Num)*Toa()+ToBij()*L16
    x12=SelectPdf(Num)*Toa()+ToBij()*L16
    x13=SelectPdf(Num)*Toa()+ToBij()*L17
    x14=SelectPdf(Num)*Toa()+ToBij()*L17
    x15=SelectPdf(Num)*Toa()+ToBij()*L18
    x16=SelectPdf(Num)*Toa()+ToBij()*L18

    x17=SelectPdf(Num)*Toa()+ToBij()*L19
    x18=SelectPdf(Num)*Toa()+ToBij()*L19
    x19=SelectPdf(Num)*Toa()+ToBij()*L20
    x20=SelectPdf(Num)*Toa()+ToBij()*L20
    x21=SelectPdf(Num)*Toa()+ToBij()*L21
    x22=SelectPdf(Num)*Toa()+ToBij()*L21
    x23=SelectPdf(Num)*Toa()+ToBij()*L22
    x24=SelectPdf(Num)*Toa()+ToBij()*L22



    data = pd.DataFrame(np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24]).T,columns=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18','x19','x20','x21','x22','x23','x24'])

    data = (data-data.mean())/data.std()
    #data = data-data.mean()
    return data















def dep5old(Num=3000):
    L1=SelectPdf(Num)

    L2=SelectPdf(Num)*Toa()+ToBij()*L1
    L3=SelectPdf(Num)*Toa()+ToBij()*L1

    L4=SelectPdf(Num)*Toa()+ToBij()*L2
    L5=SelectPdf(Num)*Toa()+ToBij()*L2
    L6=SelectPdf(Num)*Toa()+ToBij()*L3
    L7=SelectPdf(Num)*Toa()+ToBij()*L3


    L8=SelectPdf(Num)*Toa()+ToBij()*L4
    L9=SelectPdf(Num)*Toa()+ToBij()*L4
    L10=SelectPdf(Num)*Toa()+ToBij()*L5
    L11=SelectPdf(Num)*Toa()+ToBij()*L5
    L12=SelectPdf(Num)*Toa()+ToBij()*L6
    L13=SelectPdf(Num)*Toa()+ToBij()*L6
    L14=SelectPdf(Num)*Toa()+ToBij()*L7
    L15=SelectPdf(Num)*Toa()+ToBij()*L7

    x1=SelectPdf(Num)*Toa()+ToBij()*L8
    x2=SelectPdf(Num)*Toa()+ToBij()*L8
    x3=SelectPdf(Num)*Toa()+ToBij()*L9
    x4=SelectPdf(Num)*Toa()+ToBij()*L9
    x5=SelectPdf(Num)*Toa()+ToBij()*L10
    x6=SelectPdf(Num)*Toa()+ToBij()*L10
    x7=SelectPdf(Num)*Toa()+ToBij()*L11
    x8=SelectPdf(Num)*Toa()+ToBij()*L11
    x9=SelectPdf(Num)*Toa()+ToBij()*L12
    x10=SelectPdf(Num)*Toa()+ToBij()*L12
    x11=SelectPdf(Num)*Toa()+ToBij()*L13
    x12=SelectPdf(Num)*Toa()+ToBij()*L13
    x13=SelectPdf(Num)*Toa()+ToBij()*L14
    x14=SelectPdf(Num)*Toa()+ToBij()*L14
    x15=SelectPdf(Num)*Toa()+ToBij()*L15
    x16=SelectPdf(Num)*Toa()+ToBij()*L15


    data = pd.DataFrame(np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16]).T,columns=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16'])

    data = (data-data.mean())/data.std()
    #data = data-data.mean()
    return data





def dep6(Num=3000):
    L1=SelectPdf(Num)

    L2=SelectPdf(Num)*Toa()+ToBij()*L1
    L3=SelectPdf(Num)*Toa()+ToBij()*L1

    L4=SelectPdf(Num)*Toa()+ToBij()*L2
    L5=SelectPdf(Num)*Toa()+ToBij()*L2
    L6=SelectPdf(Num)*Toa()+ToBij()*L3
    L7=SelectPdf(Num)*Toa()+ToBij()*L3


    L8=SelectPdf(Num)*Toa()+ToBij()*L4
    L9=SelectPdf(Num)*Toa()+ToBij()*L4
    L10=SelectPdf(Num)*Toa()+ToBij()*L5
    L11=SelectPdf(Num)*Toa()+ToBij()*L5
    L12=SelectPdf(Num)*Toa()+ToBij()*L6
    L13=SelectPdf(Num)*Toa()+ToBij()*L6
    L14=SelectPdf(Num)*Toa()+ToBij()*L7
    L15=SelectPdf(Num)*Toa()+ToBij()*L7

    L16=SelectPdf(Num)*Toa()+ToBij()*L8
    L17=SelectPdf(Num)*Toa()+ToBij()*L8
    L18=SelectPdf(Num)*Toa()+ToBij()*L9
    L19=SelectPdf(Num)*Toa()+ToBij()*L9
    L20=SelectPdf(Num)*Toa()+ToBij()*L10
    L21=SelectPdf(Num)*Toa()+ToBij()*L10
    L22=SelectPdf(Num)*Toa()+ToBij()*L11
    L23=SelectPdf(Num)*Toa()+ToBij()*L11
    L24=SelectPdf(Num)*Toa()+ToBij()*L12
    L25=SelectPdf(Num)*Toa()+ToBij()*L12
    L26=SelectPdf(Num)*Toa()+ToBij()*L13
    L27=SelectPdf(Num)*Toa()+ToBij()*L13
    L28=SelectPdf(Num)*Toa()+ToBij()*L14
    L29=SelectPdf(Num)*Toa()+ToBij()*L14
    L30=SelectPdf(Num)*Toa()+ToBij()*L15
    L31=SelectPdf(Num)*Toa()+ToBij()*L15

    x1=SelectPdf(Num)*Toa()+ToBij()*L16
    x2=SelectPdf(Num)*Toa()+ToBij()*L16
    x3=SelectPdf(Num)*Toa()+ToBij()*L17
    x4=SelectPdf(Num)*Toa()+ToBij()*L17
    x5=SelectPdf(Num)*Toa()+ToBij()*L18
    x6=SelectPdf(Num)*Toa()+ToBij()*L18
    x7=SelectPdf(Num)*Toa()+ToBij()*L19
    x8=SelectPdf(Num)*Toa()+ToBij()*L19
    x9=SelectPdf(Num)*Toa()+ToBij()*L20
    x10=SelectPdf(Num)*Toa()+ToBij()*L20
    x11=SelectPdf(Num)*Toa()+ToBij()*L21
    x12=SelectPdf(Num)*Toa()+ToBij()*L21
    x13=SelectPdf(Num)*Toa()+ToBij()*L22
    x14=SelectPdf(Num)*Toa()+ToBij()*L22
    x15=SelectPdf(Num)*Toa()+ToBij()*L23
    x16=SelectPdf(Num)*Toa()+ToBij()*L23

    x17=SelectPdf(Num)*Toa()+ToBij()*L24
    x18=SelectPdf(Num)*Toa()+ToBij()*L24
    x19=SelectPdf(Num)*Toa()+ToBij()*L25
    x20=SelectPdf(Num)*Toa()+ToBij()*L25
    x21=SelectPdf(Num)*Toa()+ToBij()*L26
    x22=SelectPdf(Num)*Toa()+ToBij()*L26
    x23=SelectPdf(Num)*Toa()+ToBij()*L27
    x24=SelectPdf(Num)*Toa()+ToBij()*L27
    x25=SelectPdf(Num)*Toa()+ToBij()*L28
    x26=SelectPdf(Num)*Toa()+ToBij()*L28
    x27=SelectPdf(Num)*Toa()+ToBij()*L29
    x28=SelectPdf(Num)*Toa()+ToBij()*L29
    x29=SelectPdf(Num)*Toa()+ToBij()*L30
    x30=SelectPdf(Num)*Toa()+ToBij()*L30
    x31=SelectPdf(Num)*Toa()+ToBij()*L31
    x32=SelectPdf(Num)*Toa()+ToBij()*L31


    data = pd.DataFrame(np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30,x31,x32]).T,columns=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18','x19','x20','x21','x22','x23','x24','x25','x26','x27','x28','x29','x30','x31','x32'])

    data = (data-data.mean())/data.std()
    #data = data-data.mean()
    return data




def Fig_2(Num=3000):
    L1=SelectPdf(Num)
    L2=SelectPdf(Num)*Toa()+ToBij()*L1
    L6=SelectPdf(Num)*Toa()+ToBij()*L1
    L7=SelectPdf(Num)*Toa()+ToBij()*L1
    L4=SelectPdf(Num)*Toa()+ToBij()*L1

    L3=SelectPdf(Num)*Toa()+ToBij()*L2+ToBij()*L6

    L5=SelectPdf(Num)*Toa()+ToBij()*L7

    x1=SelectPdf(Num)*Toa()+ToBij()*L2
    x2=SelectPdf(Num)*Toa()+ToBij()*L2

    x3=SelectPdf(Num)*Toa()+ToBij()*L3
    x4=SelectPdf(Num)*Toa()+ToBij()*L3


    x5=SelectPdf(Num)*Toa()+ToBij()*L4
    x6=SelectPdf(Num)*Toa()+ToBij()*L4


    x7=SelectPdf(Num)*Toa()+ToBij()*L5
    x8=SelectPdf(Num)*Toa()+ToBij()*L5

    data = pd.DataFrame(np.array([x1,x2,x3,x4,x5,x6,x7,x8]).T,columns=['x1','x2','x3','x4','x5','x6','x7','x8'])

    data = (data-data.mean())/data.std()
    #data = data-data.mean()
    return data




def Case1(Num=3000):
    L1=SelectPdf(Num)
    L2=SelectPdf(Num)*Toa()+ToBij()*L1
    L3=SelectPdf(Num)*Toa()+ToBij()*L1+ToBij()*L2

    x1=SelectPdf2(Num)*Toa()+ToBij()*L1
    x2=SelectPdf2(Num)*Toa()+ToBij()*L1
    x3=SelectPdf2(Num)*Toa()+ToBij()*L1


    x4=SelectPdf2(Num)*Toa()+ToBij()*L2
    x5=SelectPdf2(Num)*Toa()+ToBij()*L2
    x6=SelectPdf2(Num)*Toa()+ToBij()*L2



    x7=SelectPdf2(Num)*Toa()+ToBij()*L3
    x8=SelectPdf2(Num)*Toa()+ToBij()*L3
    x9=SelectPdf2(Num)*Toa()+ToBij()*L3



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


    x1=SelectPdf2(Num)*Toa()+ToBij()*L1
    x2=SelectPdf2(Num)*Toa()+ToBij()*L1

    x3=SelectPdf2(Num)*Toa()+ToBij()*L2
    x4=SelectPdf2(Num)*Toa()+ToBij()*L3
    x5=SelectPdf2(Num)*Toa()+ToBij()*L4

    x6=SelectPdf2(Num)*Toa()+ToBij()*L5
    x7=SelectPdf2(Num)*Toa()+ToBij()*L5
    data = pd.DataFrame(np.array([x1,x2,x3,x4,x5,x6,x7]).T,columns=['x1','x2','x3','x4','x5','x6','x7'])

    data = (data-data.mean())/data.std()
    #data = data-data.mean()
    return data




def Case3(Num=3000):
    L1=SelectPdf(Num)
    L2=SelectPdf(Num)*Toa()+ToBij()*L1
    L3=SelectPdf(Num)*Toa()+ToBij()*L1
    L4=SelectPdf(Num)*Toa()+ToBij()*L1

    x1=SelectPdf2(Num)*Toa()+ToBij()*L1
    x2=SelectPdf2(Num)*Toa()+ToBij()*L1


    x3=SelectPdf2(Num)*Toa()+ToBij()*L2
    x4=SelectPdf2(Num)*Toa()+ToBij()*L2


    x5=SelectPdf2(Num)*Toa()+ToBij()*L3
    x6=SelectPdf2(Num)*Toa()+ToBij()*L3


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


    x1=SelectPdf2(Num)*Toa()+ToBij()*L4
    x2=SelectPdf2(Num)*Toa()+ToBij()*L4

    x3=SelectPdf2(Num)*Toa()+ToBij()*L5
    x4=SelectPdf2(Num)*Toa()+ToBij()*L5


    x5=SelectPdf2(Num)*Toa()+ToBij()*L6
    x6=SelectPdf2(Num)*Toa()+ToBij()*L6

    x7=SelectPdf2(Num)*Toa()+ToBij()*L7
    x8=SelectPdf2(Num)*Toa()+ToBij()*L7

    x9=SelectPdf2(Num)*Toa()+ToBij()*L8
    x10=SelectPdf2(Num)*Toa()+ToBij()*L8

    x11=SelectPdf2(Num)*Toa()+ToBij()*L9
    x12=SelectPdf2(Num)*Toa()+ToBij()*L9
    x13=SelectPdf2(Num)*Toa()+ToBij()*L9

    x14=SelectPdf2(Num)*Toa()+ToBij()*L2
    x15=SelectPdf2(Num)*Toa()+ToBij()*L3



    data = pd.DataFrame(np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15]).T,columns=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15'])

    data = (data-data.mean())/data.std()
    #data = data-data.mean()
    return data


def main():
    Case4()

if __name__ == '__main__':
    main()
