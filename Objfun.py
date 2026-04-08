import numpy as np
from Evaluation import net_evaluation
from Global_Vars import Global_Vars
from Model_SPN import Model_SPN


def objfun(Soln):
    Data = Global_Vars.Data
    GT = Global_Vars.GT
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = Soln[i, :]
            Images, results = Model_SPN(Data, GT, sol)
            Eval = net_evaluation(Images, results)
            Fitn[i] = 1 / Eval[6]
        return Fitn
    else:
        sol = Soln
        Images, results = Model_SPN(Data, GT, sol)
        Eval = net_evaluation(Images, results)
        Fitn = 1 / Eval[6]
        return Fitn


