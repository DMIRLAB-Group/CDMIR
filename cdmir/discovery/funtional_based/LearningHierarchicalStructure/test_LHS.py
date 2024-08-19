import Causal_Discovery_in_LHM as LHM
import Paper_simulation as SimulationData




def main():
    alpha=0.005
    data=SimulationData.CaseII(10000)
    LHM.Causal_Discovery_LHM(data, alpha)

if __name__ == '__main__':
    main()
