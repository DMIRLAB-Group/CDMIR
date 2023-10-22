import Main_LiNG_LHM
import SimulationData as SD #simulate data


#
def main():
    #generate simulation data, Paper.Case 1, can use SD.Case2 to test the Paper.Case 2
    data=SD.Case1(10000)
    #set the alpha value
    alpha=0.01
    #Esimate causal structure from observed data
    Adj_Matrix=Main_LiNG_LHM.LHME(data,alpha)
    #return the adj matrix of structure
    print('Adj_matrix',Adj_Matrix)


if __name__ == '__main__':
    main()
