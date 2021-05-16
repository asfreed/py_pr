from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
AutoMinorLocator)
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-3, 3, 9)
y1 = [i**3 for i in x]
y2 = np.abs(x)
np.random.seed(19391240)
mu, sigma = 100, 15
x1 = mu + sigma * np.random.randn(10000)

figure = plt.figure()
ax1 = figure.add_subplot(2, 2, 1)
ax2 = figure.add_subplot(2, 2, 2)
ax3 = figure.add_subplot(2, 2, 3)

ax1.set_xlim(-3, 3)
ax1.set_ylim(-27, 27)
ax1.plot(x, y1, "-b", label="y=x^3")
ax1.grid(which="major", lw=1.2)
ax1.grid(which="minor", c="lightgray", ls="--", lw=0.3)
ax1.legend()
ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax1.yaxis.set_minor_locator(AutoMinorLocator())
ax1.tick_params(which='major', length=8, width=2)
ax1.tick_params(which='minor', length=4, width=1)

ax3.hist(x1, 50, density=True, facecolor="g", alpha=0.75, label="hist")
ax3.grid(c="gray", lw=0.5)
ax3.legend()

ax2.fill_between(x, y2, color="blue", alpha=0.2)
ax2.plot(x, y2, c="slateblue", alpha=0.6, label="y2=|x|")
ax2.grid(ls="--", lw=0.4)
ax2.legend()

plt.show()