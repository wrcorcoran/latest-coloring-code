import math
from itertools import combinations

from tqdm import tqdm
from gurobipy import Model, GRB, quicksum

delta = int(1e30)
lmbda = 1.80653622916876721582468690030
p1, p2, p3, p4, p5, p6, p7 = 1, 1/3, 0.15326836257550522, 1/12, 0.03671039382375485, 0.0017430031982297372, 0
eta_g = 0.049837077123186322713532803164
# lmbda = 1.8089
# eta_g = 0.0469
# p1, p2, p3, p4, p5, p6, p7 = 1, 0.324, 0.154, 0.088, 0.044, 0.011, 0

# calculating cost
def maximize():
    k = lmbda * delta
    eta = eta_g / delta

    model = Model("maximize_score")
    model.setParam("OutputFlag", 0)
    model.setParam("BarConvTol", 1e-30)

    d_names = [
        "d_1_0", "d_1_1", "d_2_00", "d_2_10", "d_2_01", "d_2_11",
        "d_1_2", "d_2_20", "d_2_21", "d_2_22"
    ]

    d_vars = {name: model.addVar(lb=0, ub=1, name=name) for name in d_names}

    model.addConstr(quicksum(d_vars.values()) == 1, "Sum_to_1")
    # model.addConstr(d_vars["d_1_1"] == 1) # DEBUGGING
    # model.addConstr(d_vars["d_1_2"] == 1) # DEBUGGING
    model.addConstr(d_vars["d_2_22"] == 1) # DEBUGGING

    pr_ev = (
        lmbda
        - 1
        + (2 * p2 * d_vars["d_1_0"] + (p2 + p3) * d_vars["d_1_1"] + (p2 + p4) * d_vars["d_1_2"])
        + 0.5 * (
            2 * p3 * d_vars["d_2_00"]
            + (p4 + p3) * d_vars["d_2_10"]
            + (p4 + p3) * d_vars["d_2_01"]
            + 2 * p4 * d_vars["d_2_11"]
        )
    )

    # eta_cost = eta * (d_vars["d_1_0"] + d_vars["d_2_00"] + 0.5 * (d_vars["d_2_10"] + d_vars["d_2_01"]))
    eta_cost = eta_g * (d_vars["d_1_0"] + d_vars["d_2_00"] + 0.5 * (d_vars["d_2_10"] + d_vars["d_2_01"]))
    prob_term = pr_ev * eta_cost

    lambda_1_0 = 1 - p2 + eta * (1 + p2) * delta
    lambda_1_1 = 1 - p3 - eta * (k - delta - 2)
    lambda_1_2 = 1 - p2 + 2 * (p3 - p4)
    lambda_2_00 = 2 + p3 + 2 * eta * delta * (1 + p2)
    lambda_2_10 = 2 + p2 - eta * (k - (2 + p2) * delta - 2)
    lambda_2_11 = max(
        1 + 3 * p2 - p4 - 2 * eta * (k - delta - 2),
        2 * p2 - 2 * p3 + 3 * p4 - 3 * p5 - 2 * eta * (k - delta - 2),
    )
    lambda_2_20 = 2 + 2 * p3 - p5 + eta * (1 + p2) * delta
    lambda_2_21 = max(
        2 + p2 + 2 * p3 - p5 - eta * (k - delta - 2),
        2 + 3 * p4 - 2 * p6 - eta * (k - delta - 2),
    )
    lambda_2_22 = 2 + 4 * p3 + p7

    lambda_sum = (
        d_vars["d_1_0"] * lambda_1_0 + d_vars["d_1_1"] * lambda_1_1 + d_vars["d_1_2"] * lambda_1_2
        + 0.5 * (
            d_vars["d_2_00"] * (lambda_2_00 - 1)
            + d_vars["d_2_10"] * (lambda_2_10 - 1)
            + d_vars["d_2_01"] * (lambda_2_10 - 1)
            + d_vars["d_2_11"] * (lambda_2_11 - 1)
            + d_vars["d_2_20"] * (lambda_2_20 - 1)
            + d_vars["d_2_21"] * (lambda_2_21 - 1)
            + d_vars["d_2_22"] * (lambda_2_22 - 1)
        )
    )

    lambda_obj = model.addVar(lb=1, ub=2, name="lambda_obj")
    model.setObjective(lambda_obj, GRB.MAXIMIZE)
    model.addConstr(lambda_obj - (1 + ((prob_term + lambda_sum))) <= 0, "LambdaBound")

    model.optimize()
    model.write("tight_cases.lp")

    d_values = {name: var.X for name, var in d_vars.items()}
    lambda_value = lambda_obj.X

    return lambda_value, d_values

def main():
    lambda_val, d_values = maximize()

    print(f"Lambda value: {lambda_val}")
    print(f"Worst configurations: {d_values}")

if __name__ == "__main__":
    main()