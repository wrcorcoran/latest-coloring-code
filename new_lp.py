from itertools import combinations
from pulp import *
from scipy.optimize import minimize_scalar

solver = PULP_CBC_CMD(msg=False, options=["feasibilitypump"])

# delta = 250
NMAX = 7
# lmbda = 1.8089
p_vars = None
lambda_obj = None

def p(i):
    if i in range(1, NMAX + 1):
        return p_vars[i - 1]
    else:
        return 0

def stars_and_bars(n, k):
    total_len = n + k - 1
    combs = combinations(range(total_len), k - 1)
    for bars in combs:
        stars = []
        prev = -1
        for b in bars:
            stars.append(b - prev - 1)
            prev = b
        stars.append(total_len - 1 - bars[-1])
        yield (stars)


# calculating cost
def add_constraints(prob, eta):
    constraints = []
    prob += (p(1) == 1)
    prob += (p(2) <= 1/3)
    prob += (p(7) == 0)

    for j in range(3, NMAX + 1):
        prob += (p(j) <= (2/3) * p(j - 1))

    prob += (1 - p(2) >= p(2) - p(3))
    prob += (p(2) - p(3) >= 2 * (p(3) - p(4)))

    for j in range(4, NMAX):
        prob += (2 * (p(3) - p(4)) >= (j - 1) * (p(j) - p(j + 1)))

    prob += (p(4) - p(5) >= p(5) - p(6))
    prob += (p(5) - p(6) >= p(6) - p(7))

    prob += (1/2 >= p(2) + p(3))

    # # FP5
    # prob += (2 * p(2) <= 1 - 4 * p(4))
    # # FP6
    # prob += (2 * p(3) <= 4 * p(4) - p(5))

    prob += ((6/19) * p(2) - eta >= 0)

    for j in range(2, NMAX):
        prob += (p(j) <= p(j - 1))

def compute_max(delta, eta, print_constraints=False):
    global lambda_obj, p_vars
    num_vars = 10
    prob = LpProblem("Update_Coloring", LpMinimize)
    lambda_obj = LpVariable("lambda_obj", lowBound=1, upBound=2)
    p_vars = [
        LpVariable(f"p{i}", lowBound=1 if i == 1 else 0, upBound=1)
        for i in range(1, NMAX + 1)
    ]
    add_constraints(prob, eta)
    prob += lambda_obj, "Objective: Minize LambdaObj"

    for i in range(num_vars):
        combo = tuple([delta if j == i  else 0 for j in range(num_vars)])
        _ = score_5(combo, delta, prob, eta_guess_=eta)

    prob.solve(solver)
    prob.writeLP("new_lp.lp")

    new_p_values = {var.name: var.varValue for var in p_vars} 
    lambda_value = lambda_obj.varValue

    if print_constraints:   
        for i in range(num_vars):
            combo = tuple([delta if j == i  else 0 for j in range(num_vars)])
            print(i, score_5_cost(combo, delta, prob, eta_guess_=eta))  

    del prob

    return lambda_value, new_p_values

def score_5_cost(combo, delta, prob, print_=False, eta_guess_=None):
    lmbda = value(prob.objective)
    p_values = {var.name: var.varValue for var in prob.variables() if var.name.startswith("p")}
    p1, p2, p3, p4, p5, p6, p7 = [p_values[f"p{i + 1}"] for i in range(7)]

    k = lmbda * delta
    if eta_guess_ is not None:
        eta = eta_guess_ / delta
    else:
        eta = 0.049219 / delta

    # split data
    d_1_0, d_1_1, d_2_00, d_2_10, d_2_01, d_2_11, d_1_2, d_2_20, d_2_21, d_2_22 = combo

    # equation 19
    pr_ev = (
        k
        - delta
        + (2 * p2 * d_1_0 + (p2 + p3) * d_1_1 + (p2 + p4) * d_1_2)
        + (1 / 2)
        * (2 * p3 * d_2_00 + (p4 + p3) * d_2_10 + (p4 + p3) * d_2_01 + 2 * p4 * d_2_11)
    )

    eta_cost = eta * (d_1_0 + d_2_00 + (1 / 2) * (d_2_10 + d_2_01))

    # equation 20
    prob_term = pr_ev * eta_cost

    # lambdas from 13
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

    lambda_sum = (1 / 1) * (
        d_1_0 * (lambda_1_0) + d_1_1 * (lambda_1_1) + d_1_2 * (lambda_1_2)
    ) + (1 / 2) * (
        d_2_00 * (-1 + lambda_2_00)
        + d_2_10 * (-1 + lambda_2_10)
        + d_2_01 * (-1 + lambda_2_10)
        + d_2_11 * (-1 + lambda_2_11)
        + d_2_20 * (-1 + lambda_2_20)
        + d_2_21 * (-1 + lambda_2_21)
        + d_2_22 * (-1 + lambda_2_22)
    )

    if print_:
        print(prob_term, lambda_sum)

    # return (prob_term + lambda_sum)
    return 1 + ((prob_term + lambda_sum) / delta)

def score_5(combo, delta, prob, eta_guess_=None):
    k = lambda_obj * delta
    if eta_guess_ is not None:
        eta = eta_guess_ / delta
    else:
        eta = 0.0469 / delta

    non_zero_ind = combo.index(next(filter(lambda x: x != 0, combo)))

    # split data
    d_1_0, d_1_1, d_2_00, d_2_10, d_2_01, d_2_11, d_1_2, d_2_20, d_2_21, d_2_22 = combo

    # equation 19
    pr_ev = LpVariable(f"pr_ev_{non_zero_ind}", lowBound=0)
    pr_ev_expr = (
        k
        - delta
        + (2 * p(2) * d_1_0 + (p(2) + p(3)) * d_1_1 + (p(2) + p(4)) * d_1_2)
        + (1 / 2)
        * (2 * p(3) * d_2_00 + (p(4) + p(3)) * d_2_10 + (p(4) + p(3)) * d_2_01 + 2 * p(4) * d_2_11)
    )
    prob += pr_ev >= pr_ev_expr

    eta_cost = eta * (d_1_0 + d_2_00 + (1 / 2) * (d_2_10 + d_2_01))

    # equation 20
    prob_term = pr_ev * eta_cost

    # lambdas from 13
    lambda_1_0 = LpVariable(f"lambda_1_0_{non_zero_ind}", lowBound=0)
    prob += lambda_1_0 >= 1 - p(2) + eta * (1 + p(2)) * delta

    lambda_1_1 = LpVariable(f"lambda_1_1_{non_zero_ind}", lowBound=0)
    prob += lambda_1_1 >= 1 - p(3) - eta * (k - delta - 2)

    lambda_1_2 = LpVariable(f"lambda_1_2_{non_zero_ind}", lowBound=0)
    prob += lambda_1_2 >= 1 - p(2) + 2 * (p(3) - p(4))

    lambda_2_00 = LpVariable(f"lambda_2_00_{non_zero_ind}", lowBound=0)
    prob += lambda_2_00 >= 2 + p(3) + 2 * eta * delta * (1 + p(2))

    lambda_2_10 = LpVariable(f"lambda_2_10_{non_zero_ind}", lowBound=0)
    prob += lambda_2_10 >= 2 + p(2) - eta * (k - (2 + p(2)) * delta - 2)

    expr1_2_11 = 1 + 3 * p(2) - p(4) - 2 * eta * (k - delta - 2)
    expr2_2_11 = 2 * p(2) - 2 * p(3) + 3 * p(4) - 3 * p(5) - 2 * eta * (k - delta - 2)
    lambda_2_11 = LpVariable(f"lambda_2_11_{non_zero_ind}", lowBound=0)
    prob += lambda_2_11 >= expr1_2_11 
    prob += lambda_2_11 >= expr2_2_11 

    lambda_2_20 = LpVariable(f"lambda_2_20_{non_zero_ind}", lowBound=0)
    prob += lambda_2_20 >= 2 + 2 * p(3) - p(5) + eta * (1 + p(2)) * delta
    
    expr1 = 2 + p(2) + 2 * p(3) - p(5) - eta * (k - delta - 2)
    expr2 = 2 + 3 * p(5) - 2 * p(6) - eta * (k - delta - 2)
    lambda_2_21 = LpVariable(f"lambda_2_21_{non_zero_ind}", lowBound=0)
    prob += lambda_2_21 >= expr1
    prob += lambda_2_21 >= expr2 

    lambda_2_22 = LpVariable(f"lambda_2_22_{non_zero_ind}", lowBound=0)
    prob += lambda_2_22 >= 2 + 4 * p(3) + p(7)

    lambda_sum = (1 / 1) * (
        d_1_0 * (lambda_1_0) + d_1_1 * (lambda_1_1) + d_1_2 * (lambda_1_2)
    ) + (1 / 2) * (
        d_2_00 * (-1 + lambda_2_00)
        + d_2_10 * (-1 + lambda_2_10)
        + d_2_01 * (-1 + lambda_2_10)
        + d_2_11 * (-1 + lambda_2_11)
        + d_2_20 * (-1 + lambda_2_20)
        + d_2_21 * (-1 + lambda_2_21)
        + d_2_22 * (-1 + lambda_2_22)
    )

    score_of_i = LpVariable(f"cost_at_{non_zero_ind}")
    prob += score_of_i - (1 + ((prob_term + lambda_sum) / delta)) >= 0
    
    prob += lambda_obj - score_of_i >= 0

def main():
    delta = int(1e10)

    def objective(eta_guess):
        lambda_val, _ = compute_max(delta, eta_guess)
        return lambda_val

    result = minimize_scalar(objective, bounds=(0.0, 0.2), method='bounded', options={'xatol': 1e-8})

    if result.success:
        best_eta = result.x
        best_lambda, best_p_vals = compute_max(delta, best_eta, print_constraints=True)
        print(f"Best eta_guess: {best_eta:.12f} with lambda = {best_lambda:.12f}")
        print(f"Best P Vals: {best_p_vals}")
    else:
        print("Failed to find a minimizing eta in [0, 0.2]")

if __name__ == "__main__":
    main()