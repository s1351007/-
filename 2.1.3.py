# 第二章 2.1
# 主題：最佳化問題與目標函數
#
# Function:
# f(x) = (x - 2)^2
#
# Purpose:
# 1. Draw objective function
# 2. Identify theoretical optimum
# 3. Understand minimization
# ============================================================

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. Define objective function
# ------------------------------------------------------------

def objective(x):

    return (x - 2)**2


# ------------------------------------------------------------
# 2. Generate x values
# ------------------------------------------------------------

x = np.linspace(
    -5,
    5,
    500
)

y = objective(
    x
)


# ------------------------------------------------------------
# 3. Theoretical optimum
# ------------------------------------------------------------

x_best = 2

y_best = objective(
    x_best
)


# ------------------------------------------------------------
# 4. Draw function
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    x,
    y,
    label="f(x) = (x - 2)^2"
)

plt.scatter(
    x_best,
    y_best,
    marker="*",
    s=150,
    label="Theoretical optimum"
)

plt.xlabel(
    "x"
)

plt.ylabel(
    "f(x)"
)

plt.title(
    "Objective Function"
)

plt.legend()

plt.grid(True)

plt.show()


# ------------------------------------------------------------
# 5. Print result
# ------------------------------------------------------------

print("Theoretical optimal x =", x_best)

print("Minimum function value =", y_best)


