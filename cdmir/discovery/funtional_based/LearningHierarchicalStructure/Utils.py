'''
    A practial tools for merge, check contain...

'''


# extend the AllCausalCluster to contains all elements of Current_Clusters
def ExtendList(AllCausalCluster, Current_Clusters):
    for Ls in Current_Clusters:
        if not GeneralSetContains(Ls, AllCausalCluster):
            AllCausalCluster.append(Ls)
    return AllCausalCluster


def GetEarlyLearningByDic(S1, Dic):
    LKey = Dic.keys()
    for L in LKey:
        S2 = Dic[L]
        if set(S1) == set(S2):
            return L
    return ''


def GeneralDicContains(S1, Dic):
    LKey = Dic.keys()
    for L in LKey:
        S2 = Dic[L]
        if set(S1) == set(S2):
            return True

    return False


# general operator for a set and a list
def GeneralSetOperator(S1, Sets, Operator):
    S1 = set(S1)
    for S2 in Sets:
        if Operator == '>':
            if S1 > set(S2):
                return True
        elif Operator == '>=':
            if S1 >= set(S2):
                return True
        elif Operator == '=':
            if S1 == set(S2):
                return True
        elif Operator == '<':
            if S1 < set(S2):
                return True
        elif Operator == '<=':
            if S1 <= set(S2):
                return True
    return False


# Sets=[[1,2,3]]  S1 must be [1,2,3]
def GeneralSetContains(S1, Sets):
    S1 = set(S1)
    for S2 in Sets:
        if S1 == set(S2):
            return True

    return False


# Sets=[[1,2,3]]  S1=[1,2] or S1=[1,2,3]
def GeneralSetCover(S1, Sets):
    S1 = set(S1)
    for S2 in Sets:
        if S1 <= set(S2):
            return True
    return False


# Sets=[[1,2,3]]  S1 must be [1,2]
def GeneralSetStrictCover(S1, Sets):
    S1 = set(S1)
    for S2 in Sets:
        if S1 < set(S2):
            return True
    return False


def main():
    S1 = ['x1', 'x2']
    Sets = [['x1', 'x2'], ['x2', 'x5'], ['x4', 'x5'], ['x6', 'x7'], ['x8', 'x9']]
    print(GeneralSetOperator(S1, Sets, '>='))


if __name__ == '__main__':
    main()
