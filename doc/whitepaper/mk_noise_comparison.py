#!/usr/bin/env python3

# mk_noise-comparison.py
# plots power distribution graphs for various colored noise types

# (C) Jan 2026 <christian.tschudin@unibas.ch> HB9HUH/K6CFT
# SW released under the MIT license

import numpy as np
import matplotlib.pyplot as plt

MAXF = 2500

def color(fvect, p): # frequency weights
    # p values:
    #   2.0 brown
    #   1.0 pink
    #   0.5 lightpink
    #   0.0 white
    #  -0.5 lightblue
    #  -1.0 blue
    #  -2.0 violet
    return np.array([ 1/np.pow(f,p) for f in fvect ])

def hue(fvect, h):  # frequency weights
    if h == 'blueish':
        return np.array([ np.sin(np.pi / 2 * f / MAXF) for f in fvect ])
    if h == 'reddish':
        return np.array([ np.cos(np.pi / 2 * f / MAXF) for f in fvect ])


# --------
fig,axes = plt.subplots(2,3, figsize=(5,3.5))

x = 1 + np.arange(MAXF)

# ---------------------------------------------------------------------------
R1=[0.0000001,1]
R2=[0.9,19000000]
R3=[0.1,10]

ax = axes[0][0]
ax.loglog(x, 1e7 * np.pow(color(x, 1.),2), 'r')
ax.set_ylabel("power | loglog")
ax.grid(True, which="both")
ax.set_ylim(R2)

ax = axes[0][1]
ax.loglog(x, np.pow(color(x, 0.),2), 'r')
ax.grid(True, which="both")
ax.set_yticks([])
ax.set_ylim(R3)

ax = axes[0][2]
ax.loglog(x, np.pow(color(x, -1.),2), 'r')
ax.grid(True, which="both")
ax.set_yticks([])
ax.set_ylim(R2)

# ---------------------------------------------------------------------------
ax = axes[1][0]
ax.plot(x, np.pow(color(x, 1.),2), 'r')
ax.set_xlabel("p=1.0\npink")
ax.set_ylabel("power | linear")

ax = axes[1][1]
ax.plot(x, np.pow(color(x, 0.),2), 'r')
ax.set_xlabel("p=0.0\nwhite")
ax.set_yticks([])

ax = axes[1][2]
ax.plot(x, np.pow(color(x, -1.),2), 'r')
ax.set_yticks([])
ax.set_xlabel("p=-1.0\nblue")

# ---------------------------------------------------------------------------
plt.subplots_adjust(top=0.95, bottom=0.18, left=0.13, right=0.97)

FNAME = 'fig/noise-gallery-1'
for t in ['png', 'pdf']:
    plt.savefig(FNAME + '.' + t, format=t, dpi=100)
    print(f"--> {FNAME}.{t}")

# plt.show()



# ---------------------------------------------------------------------------
fig,axes = plt.subplots(2,3, figsize=(5,3.5))

ax = axes[0][0]
ax.plot(x, np.pow(hue(x, 'reddish'),2), 'r')
ax.set_xticks([])
ax.set_ylabel("power | linear")

ax = axes[0][1]
ax.plot(x, np.pow(color(x, 0.),2), 'r')
ax.set_yticks([])
ax.set_xticks([])

ax = axes[0][2]
ax.plot(x, np.pow(hue(x, 'blueish'),2), 'r')
ax.set_yticks([])
ax.set_xticks([])

ax = axes[1][0]
ax.plot(x, hue(x, 'reddish'), 'green')
ax.set_ylabel("weight | linear")
ax.set_xlabel("reddish", fontweight='bold')

ax = axes[1][1]
ax.plot(x, color(x, 0.), 'green')
ax.set_yticks([])
ax.set_xlabel("white")

ax = axes[1][2]
ax.plot(x, hue(x, 'blueish'), 'green')
ax.set_yticks([])
ax.set_xlabel("blueish", fontweight='bold')

# ---------------------------------------------------------------------------
plt.subplots_adjust(top=0.95, bottom=0.15, left=0.13, right=0.97)

FNAME = 'fig/noise-gallery-2'
for t in ['png', 'pdf']:
    plt.savefig(FNAME + '.' + t, format=t, dpi=100)
    print(f"--> {FNAME}.{t}")

plt.show()

# eof
