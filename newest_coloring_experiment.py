from itertools import combinations
import math
from tqdm import tqdm

# delta = 250
lmbda = 1.8089
p1, p2, p3, p4, p5, p6, p7 = 1, 0.324, 0.154, 0.088, 0.044, 0.011, 0


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
def score_5(combo, print_=False):
    k = lmbda * delta
    eta = (p2 - p3) * delta * (1 / 2) * (1 / k)

    # split data
    d_1_0, d_1_1, d_2_00, d_2_10, d_2_11 = combo
    d_1_2, d_2_20, d_2_21, d_2_22 = 0, 0, 0, 0

    # equation 20
    pr_ev = (
        k
        - delta
        + (2 * p2 * d_1_0 + (p2 + p3) * d_1_1 + (p2 + p4) * d_1_2)
        + (1 / 2) * (2 * p3 * d_2_00 + 2 * (p4 + p3) * d_2_10 + 2 * p4 * d_2_11)
    )
    eta_cost = eta * (d_1_0 + d_2_00 + (1 / 2) * (2 * d_2_10))

    prob_term = pr_ev * eta_cost

    # lambdas from 13
    claim_1_0 = 1 - p2 + eta * (1 + p2) * delta
    claim_1_1 = 1 - p3 - eta * (k - delta - 2)
    claim_1_2 = 1 - p2 + 2 * (p3 - p4)
    claim_2_00 = 2 + p3 + 2 * eta * delta * (1 + p2)
    claim_2_10 = 2 + p2 - eta * (k - (2 + p2) * delta - 2)
    claim_2_11 = max(
        1 + 3 * p2 - p4 - 2 * eta * (k - delta - 2),
        2 * p2 - 2 * p3 + 3 * p4 - 3 * p5 - 2 * eta * (k - delta - 2),
    )
    claim_2_20 = 2 + 2 * p3 - p5 + eta * (1 + p2) * delta
    claim_2_21 = max(
        2 + p2 + 2 * p3 - p5 - eta * (k - delta - 2),
        2 + 3 * p4 - 2 * p6 - eta * (k - delta - 2),
    )
    claim_2_22 = 2 + 4 * p3 + p7

    lambda_term = (1 / 1) * (
        d_1_0 * (1 + claim_1_0) + d_1_1 * (1 + claim_1_1) + d_1_2 * (1 + claim_1_2)
    ) + (1 / 2) * (
        d_2_00 * (1 + claim_2_00)
        + d_2_10 * (1 + claim_2_10)
        + d_2_11 * (1 + claim_2_11)
        + d_2_20 * (1 + claim_2_20)
        + d_2_21 * (1 + claim_2_21)
        + d_2_22 * (1 + claim_2_22)
    )

    if print_:
        print(prob_term, lambda_term)

    return prob_term + lambda_term


if __name__ == "__main__":
    count = 0
    delta = int(input("Please enter a value for delta: "))
    max_combo, max_val = (), -float("inf")
    num_vars = 5
    total_itrs = math.comb(delta + num_vars - 1, num_vars - 1)
    for combo in tqdm(
        stars_and_bars(delta, num_vars), total=total_itrs, desc="Processing"
    ):
        score = score_5(combo)
        if score > max_val:
            max_combo, max_val = combo, score
        count += 1

    print(f"Total combinations: {count}")
    print(f"Max Combo: {max_combo}")
    print(f"Max Val: {max_val}")
