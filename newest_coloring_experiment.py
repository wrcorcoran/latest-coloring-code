import math
from itertools import combinations

from tqdm import tqdm

# delta = 250
lmbda = 1.806858
p1, p2, p3, p4, p5, p6, p7 = 1, 1/3, 0.1534, 1/12, 0.0366, 0.00155, 0


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
def score_5(combo, delta, print_=False, eta_guess_=None):
    k = lmbda * delta
    if eta_guess_ is not None:
        eta = eta_guess_ / delta
    else:
        # eta = eta_guess_ * (1 / 2) * (1 / k)
        eta = 0.049219 / delta
        # eta = 0.047357 / delta
        # eta = 0.05 / delta

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

def main():
    mode = input("Choose mode: [1] Max Combo Search, [2] Eta Binary Search, [3] Max Over (Presumed) Extremal Cases: ").strip()

    delta = int(input("Please enter a value for delta: "))
    num_vars = 10

    if mode == "1":
        count = 0
        max_combo, max_val = (), -float("inf")
        total_itrs = math.comb(delta + num_vars - 1, num_vars - 1)

        for combo in tqdm(stars_and_bars(delta, num_vars), total=total_itrs, desc="Processing"):
            score = score_5(combo, delta, eta_guess_=0.008)
            if score > max_val:
                max_combo, max_val = combo, score
            count += 1

        print(f"Total combinations: {count}")
        print(f"Max Combo: {max_combo}")
        print(f"Max Val: {max_val}")

    elif mode == "2":
        combo1 = [delta if i == 0 else 0 for i in range(num_vars)]
        combo2 = [delta if i == 1 else 0 for i in range(num_vars)]

        print("Combo1:", combo1)
        print("Combo2:", combo2)

        low = 0.0
        high = 1.0
        threshold = 1e-6
        best_eta = None

        while high - low > threshold:
            mid = (low + high) / 2
            eta_guess = mid

            score1 = score_5(tuple(combo1), delta, False, eta_guess)
            score2 = score_5(tuple(combo2), delta, False, eta_guess)

            if score1 > score2:
                best_eta = eta_guess
                high = mid
            else:
                low = mid

        if best_eta is not None:
            print(f"Found eta_guess: {best_eta:.6f} where score1 > score2")
        else:
            print("No eta_guess in [0, 1] satisfies score1 > score2")

    elif mode == "3":
        eta = 0.049219
        num_vars = 10
        max_comb, max_v = [], 0
        for i in range(num_vars):
            combo = tuple([delta if j == i  else 0 for j in range(num_vars)])
            val = score_5(combo, delta, eta_guess_=eta)
            if val > max_v:
                max_comb = combo
                max_v = val

        print(max_comb)

    else:
        print("Invalid mode selected.")

if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     count = 0
#     delta = int(input("Please enter a value for delta: "))
#     max_combo, max_val = (), -float("inf")
#     # num_vars = 5
#     num_vars = 10
#     total_itrs = math.comb(delta + num_vars - 1, num_vars - 1)
#     for combo in tqdm(
#         stars_and_bars(delta, num_vars), total=total_itrs, desc="Processing"
#     ):
#         score = score_5(combo)
#         if score > max_val:
#             max_combo, max_val = combo, score
#         count += 1

#     print(f"Total combinations: {count}")
#     print(f"Max Combo: {max_combo}")
#     print(f"Max Val: {max_val}")

#     # for i in range(num_vars):
#     #     arr = [0 for _ in range(num_vars)]
#     #     arr[i] = delta
#     #     score = score_5(tuple(arr))
#     #     print(f"Val: {score}")

# # if __name__ == "__main__":
# #     count = 0
# #     delta = int(input("Please enter a value for delta: "))
# #     num_vars = 10

# #     combo1 = [delta if i == 0 else 0 for i in range(num_vars)]
# #     combo2 = [delta if i == 1 else 0 for i in range(num_vars)]
# #     print("Combo1:", combo1)
# #     print("Combo2:", combo2)

# #     low = 0.0
# #     high = 1
# #     threshold = 1e-6
# #     best_eta = None

# #     while high - low > threshold:
# #         mid = (low + high) / 2
# #         eta_guess = mid

# #         score1 = score_5(tuple(combo1), False, eta_guess)
# #         score2 = score_5(tuple(combo2), False, eta_guess)

# #         if score1 > score2:
# #             best_eta = eta_guess
# #             high = mid
# #         else:
# #             low = mid

# #     if best_eta is not None:
# #         print(f"Found eta_guess: {best_eta:.6f} where score1 > score2")
# #     else:
# #         print("No eta_guess in [0, 0.1] satisfies score1 > score2")

# # print(f"Current: {p2 - p3}")
