import numpy as np
import matplotlib.pyplot as plt

N = 100
x = np.linspace(0, 1, N)
m, c = 5.5, 15.
err = 1.
y = m * x + c + np.random.randn(N)*err
yerr = np.ones_like(y)*err


def fit_line(x, y, yerr):
    AT = np.vstack((np.ones(len(x)), x))
    C = np.eye(len(x))*yerr
    CA = np.linalg.solve(C, AT.T)
    Cy = np.linalg.solve(C, y)
    ATCA = np.dot(AT, CA)
    ATCy = np.dot(AT, Cy)
    w = np.linalg.solve(ATCA, ATCy)

    cov = np.linalg.inv(ATCA)
    sig = np.sqrt(np.diag(cov))
    return w, sig

w, sig = fit_line(x, y, yerr)
print(f"{w[0]:.2f}, +/- {sig[0]:.1f}, {w[1]:.2f}, +/- {sig[1]:.1f}")

plt.errorbar(x, y, yerr=yerr, fmt="k.", ms=1, lw=.5, alpha=.5)
plt.fill_between(x, x*(w[1]+sig[1])+(w[0]+sig[0]),
                 x*(w[1]-sig[1])+(w[0]-sig[0]), alpha=.5)
plt.savefig("test")

# p, cov = np.polyfit(x, y, 1, cov=True)
# plt.plot(x, np.polyval(p, x), "C1", lw=3)
# plt.plot(x, np.polyval([m, c], x), "C4")

# m, c = p
# merr, cerr = np.sqrt(np.diag(cov))
