import numpy as np
import math
import scipy
import matplotlib.pyplot as plt

params = {'text.usetex': True,
          'font.size': 10,
          'text.latex.preamble': r'\usepackage{amsmath}'}
plt.rcParams.update(params)

def gen_semicircle_top(r, h, k):
    x0 = h - r  # determine x start
    x1 = h + r  # determine x finish
    x = np.linspace(x0, x1, 10000)  # many points to solve for y

    # use numpy for array solving of the semicircle equation
    y = k + np.sqrt(r**2 - (x - h)**2)
    return x, y

def gen_semicircle_bottom(r, h, k):
    x0 = h - r  # determine x start
    x1 = h + r  # determine x finish
    x = np.linspace(x0, x1, 10000)  # many points to solve for y

    # use numpy for array solving of the semicircle equation
    y = k + np.sqrt(r**2 - (x - h)**2)
    return x, -y

def gen_line(x_0, x_1, y_0, y_1):
    x = np.linspace(x_0, x_1, 1000)
    y = y_0 + (y_1-y_0)/(x_1-x_0) * (x-x_0)
    return x, y

x=np.linspace(-10,10,2000)

fig, ax1 = plt.subplots(figsize=(3,3))

x1,y1 = gen_semicircle_top(2.75,0,0)
x2,y2 = gen_semicircle_top(0.75,0,0)
x3,y3 = gen_line(-2.75, -0.75, 0, 0)
x4,y4 = gen_line(0.75, 2.75, 0, 0)

ax1.spines['left'].set_position(('data', 0))
ax1.spines['bottom'].set_position(('data', 0))
ax1.spines['right'].set_color('none')
ax1.spines['top'].set_color('none')
ax1.xaxis.set_ticks_position('none')
ax1.yaxis.set_ticks_position('none')
ax1.axes.xaxis.set_ticklabels([])
ax1.axes.yaxis.set_ticklabels([])

ax1.set_xlabel(r"$\mathrm{Re}$", loc='right')
ax1.set_ylabel(r"$\mathrm{Im}$", loc='top', rotation='horizontal')
ax1.set_xlim(-3.5, 3.5)
ax1.set_ylim(-1.5, 3.5)

ax1.plot(x1, y1, color='k')
ax1.plot(x2, y2, color='k')
ax1.plot(x3, y3, color='k')
ax1.plot(x4, y4, color='k')


ax1.plot(0, -1, marker='x', color='k', ms=5)
ax1.text(0.2, -0.9, r'$\text{i}\nu_{-1}$')

ax1.plot(0, 0, marker='x', color='k', ms=5)
ax1.text(0.2, 0.1, r'$\text{i}\nu_0$')

ax1.plot(0, 1, marker='x', color='k', ms=5)
ax1.text(0.2, 1.1, r'$\text{i}\nu_1$')

ax1.plot(0, 2, marker='x', color='k', ms=5)
ax1.text(0.2, 2.1, r'$\text{i}\nu_2$')

ax1.plot(0, 3, marker='x', color='k', ms=5)
ax1.text(0.2, 3.1, r'$\text{i}\nu_3$')

ax1.plot(0, -1, marker='x', color='k', ms=5)
ax1.text(0.2, -0.9, r'$\text{i}\nu_{-1}$')


ax1.text(0.6, -0.4, r'$r$')
ax1.text(-1, -0.4, r'$-r$')
ax1.text(2.6, -0.4, r'$R$')
ax1.text(-3, -0.4, r'$-R$')



ax1.plot(1.1, 0.75, marker='x', color='k', ms=5)
ax1.text(1.3, 0.85, r'$\omega_1$')

ax1.plot(-1.1, 0.75, marker='x', color='k', ms=5)
ax1.text(-1.7, 0.85, r'$\omega_2$')

ax1.plot(-1.1, -0.75, marker='x', color='k', ms=5)
ax1.text(-1.7, -1, r'$\omega_3$')

ax1.plot(1.1, -0.75, marker='x', color='k', ms=5)
ax1.text(1.3, -1, r'$\omega_4$')


ax1.plot(-1.5, 0, marker='$>$', color='k', ms=5)
ax1.plot(0, 0.75, marker='$>$', color='k', ms=5)
ax1.plot(1.5, 0, marker='$>$', color='k', ms=5)
ax1.plot(0, 2.75, marker='$<$', color='k', ms=5)


ax1.plot((1), (0), ls="", marker=">", ms=5, color="k", transform=ax1.get_yaxis_transform(), clip_on=False)
ax1.plot((0), (1), ls="", marker="^", ms=5, color="k", transform=ax1.get_xaxis_transform(), clip_on=False)
ax1.set_aspect(1)

plt.savefig('BATH_CORRELATION_FUNCTION_INTEGRATION_CONTOUR.png', dpi=300, bbox_inches='tight')
plt.show()
