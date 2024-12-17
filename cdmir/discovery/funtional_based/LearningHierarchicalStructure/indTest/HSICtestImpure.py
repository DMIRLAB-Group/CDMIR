# -------------------------------------------------------------------------------
# Name:        模块1
# Purpose:
#
# Author:      YY
#
# Created:     25/10/2021
# Copyright:   (c) YY 2021
# Licence:     <your licence>
# -------------------------------------------------------------------------------
import LearningHierarchicalStructure.GIN2 as GIN
import numpy as np
import pandas as pd
import LearningHierarchicalStructure.Paper_simulation as SD


def main22():
    data = SD.CaseIV(20000)
    # print(data.columns)
    # X11 X4  X6->X12
    X = ['x11', 'x6', 'x12']
    Z = ['x4', 'x6']
    GIN.getomega(data, X, Z)

    X = ['x11', 'x6', 'x12']
    Z = ['x4', 'x12']

    GIN.getomega(data, X, Z)


def main():
    Num = 30000
    L = np.random.uniform(size=Num)
    L2 = np.random.uniform(size=Num) + L * ToBij()
    x1 = np.random.uniform(size=Num) * 0.2 + L * ToBij()
    x2 = np.random.uniform(size=Num) * 0.2 + L * ToBij() + x1 * ToBij()
    x3 = np.random.uniform(size=Num) * 0.2 + L2 * ToBij()
    x4 = np.random.uniform(size=Num) * 0.2 + L2 * ToBij()
    data = pd.DataFrame(np.array([x1, x2, x3, x4]).T, columns=['x1', 'x2', 'x3', 'x4'])

    X = ['x1', 'x2', 'x3']
    Z = ['x1', 'x4']

    GIN.getomega(data, X, Z)

    X = ['x1', 'x2', 'x3']
    Z = ['x2', 'x4']
    GIN.getomega(data, X, Z)


def ToBij():
    ten = np.random.randint(0, 2)
    s = np.random.random()
    while abs(s) < 0.5 and ten == 0:
        s = np.random.random()
    result = ten + s
    if np.random.randint(0, 10) > 5:
        result = -1 * result
    return round(result, 3)


if __name__ == '__main__':
    main()
