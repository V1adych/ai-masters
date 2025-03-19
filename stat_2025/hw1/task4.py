from typing import Tuple
import numpy as np
from matplotlib import pyplot as plt


def get_estimates(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    first_moment = np.mean(data, axis=1)
    second_moment = np.mean(data * data, axis=1)

    subtrahend = np.sqrt(3) * np.sqrt(second_moment - first_moment * first_moment)
    a = first_moment - subtrahend
    b = first_moment + subtrahend
    return a, b


def main():
    ns = np.logspace(1, 5, 10, dtype=int)
    trials = 1000

    a_true = -np.exp(2)
    b_true = np.pi * 2.5

    plt.figure(figsize=(10, 6))

    errors_a = []
    errors_b = []
    for n in ns:
        samples = np.random.uniform(a_true, b_true, (trials, n))
        a_est, b_est = get_estimates(samples)
        errors_a.append(np.mean((a_est - a_true) ** 2))
        errors_b.append(np.mean((b_est - b_true) ** 2))

    plt.plot(ns, errors_a, "o-", label="Error of a")
    plt.plot(ns, errors_b, "o-", label="Error of b")
    plt.xlabel("n")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid()
    plt.yscale("log")
    plt.xscale("log")

    plt.savefig("task4.png")


if __name__ == "__main__":
    main()
