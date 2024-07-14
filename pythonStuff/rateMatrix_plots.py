import matplotlib.pyplot as plt
import numpy as np
import rateMatrix
import DDS


params = {'text.usetex': True,
          'text.latex.preamble': r'\usepackage{amsmath}'}
plt.rcParams.update(params)


smallest_nonzero_eigval_vec = np.vectorize(rateMatrix.smallest_nonzero_eigval)


x = np.linspace(0,0.375,200)


fig, ax1 = plt.subplots()

ax1.set_xlabel(r"$\epsilon$")
ax1.set_ylabel(r"$\Gamma$")
ax1.set_xlim(0, 0.375)
ax1.set_ylim(0, 0.0028)
ax1.set_xticks([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35])
ax1.set_yticks([0,0.0004,0.0008,0.0012,0.0016,0.002,0.0024,0.0028])
ax1.set_aspect(125)
ax1.plot(x,smallest_nonzero_eigval_vec(1.4, x, 1, 0.097, 0.18, 0.6), "k-", label=r"$T=0.6$", linewidth=1)
ax1.plot(x,smallest_nonzero_eigval_vec(1.4, x, 1, 0.097, 0.18, 0.5), "k--", label=r"$T=0.5$", linewidth=1)
ax1.plot(x,smallest_nonzero_eigval_vec(1.4, x, 1, 0.097, 0.18, 0.3), "k:", label=r"$T=0.3$", linewidth=1)
ax1.legend(frameon=False)
plt.savefig('rateMatrixTemp.png', dpi=300, bbox_inches='tight')


fig, ax2 = plt.subplots()

ax2.set_xlabel(r"$\epsilon$")
ax2.set_ylabel(r"$\Gamma$")
ax2.set_xlim(0, 0.375)
ax2.set_ylim(0, 0.0028)
ax2.set_xticks([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35])
ax2.set_yticks([0,0.0004,0.0008,0.0012,0.0016,0.002,0.0024,0.0028])
ax2.set_aspect(125)
ax2.plot(x,smallest_nonzero_eigval_vec(1.4, x, 1.2, 0.097, 0.18, 0.5), "k-", label=r"$\Omega=1.2$", linewidth=1)
ax2.plot(x,smallest_nonzero_eigval_vec(1.4, x, 1, 0.097, 0.18, 0.5), "k--", label=r"$\Omega=1$", linewidth=1)
ax2.plot(x,smallest_nonzero_eigval_vec(1.4, x, 0.9, 0.097, 0.18, 0.5), "k:", label=r"$\Omega=0.9$", linewidth=1)
ax2.legend(frameon=False)
plt.savefig('rateMatrixOmega.png', dpi=300, bbox_inches='tight')

plt.show()
