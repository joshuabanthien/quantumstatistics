import numpy as np
import math
import scipy
import matplotlib.pyplot as plt

params = {'text.usetex': True,
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

fig, ax1 = plt.subplots()

x1,y1 = gen_semicircle_top(6,0,0)
x2,y2 = gen_semicircle_top(1,0,0)
x3,y3 = gen_line(-6, -1, 0, 0)
x4,y4 = gen_line(1, 6, 0, 0)

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
ax1.set_xlim(-10, 10)
ax1.set_ylim(-5, 10)

ax1.plot(x1, y1, color='k')
ax1.plot(x2, y2, color='k')
ax1.plot(x3, y3, color='k')
ax1.plot(x4, y4, color='k')

ax1.plot(0, 0, marker='x', color='k', ms=5)
ax1.text(0.4, -0.5, r'$\nu_0$')
ax1.plot(0, 2.4, marker='x', color='k', ms=5)
ax1.text(0.4, 2.4, r'$\nu_1$')
ax1.plot(0, 4.8, marker='x', color='k', ms=5)
ax1.text(0.4, 4.8, r'$\nu_2$')
ax1.plot(0, 7.2, marker='x', color='k', ms=5)
ax1.text(0.4, 7.2, r'$\nu_3$')
ax1.plot(0, 9.6, marker='x', color='k', ms=5)
ax1.text(0.4, 9.6, r'$\nu_4$')
ax1.plot(0, -2.4, marker='x', color='k', ms=5)
ax1.text(0.4, -2.9, r'$\nu_{-1}$')
ax1.plot(0, -4.8, marker='x', color='k', ms=5)
ax1.text(0.4, -5.3, r'$\nu_{-2}$')
#ax1.plot(0, -7.2, marker='x', color='k', ms=5)
#ax1.text(0.4, -7.7, r'$\nu_{-3}$')
#ax1.plot(0, -9.6, marker='x', color='k', ms=5)
#ax1.text(0.4, -10.1, r'$\nu_{-4}$')

ax1.plot(2.4, 3.5, marker='x', color='k', ms=5)
ax1.text(2.8, 3.5, r'$\omega_1$')
ax1.plot(-2.4, 3.5, marker='x', color='k', ms=5)
ax1.text(-3.4, 3.5, r'$\omega_2$')
ax1.plot(-2.4, -3.5, marker='x', color='k', ms=5)
ax1.text(-3.4, -4, r'$\omega_3$')
ax1.plot(2.4, -3.5, marker='x', color='k', ms=5)
ax1.text(2.8, -4, r'$\omega_4$')

ax1.plot(-3.5, 0, marker='>', color='k', ms=5)
ax1.plot(0, 1, marker='>', color='k', ms=5)
ax1.plot(3.5, 0, marker='>', color='k', ms=5)
ax1.plot(0, 6, marker='<', color='k', ms=5)

ax1.arrow(0, 0, -0.62, 0.62, color='red', width=0.001, length_includes_head='True', head_width=0.1)
ax1.arrow(0, 0, 4.15, 4.15, color='red', width=0.001, length_includes_head='True', head_width=0.1)
ax1.text(-1, 0.7071, r'$r$', color='red')
ax1.text(4.35, 4.242, r'$R$', color='red')

ax1.plot((1), (0), ls="", marker=">", ms=3, color="k", transform=ax1.get_yaxis_transform(), clip_on=False)
ax1.plot((0), (1), ls="", marker="^", ms=3, color="k", transform=ax1.get_xaxis_transform(), clip_on=False)
ax1.set_aspect(1)

plt.savefig('intContourPlot.png', dpi=300, bbox_inches='tight')
plt.show()
