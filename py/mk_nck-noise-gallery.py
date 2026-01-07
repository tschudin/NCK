#!/usr/bin/env python3

# mk_nck-noise-gallery.py
# plots power distribution graphs for various colored noise types

# (C) Jan 2026 <christian.tschudin@unibas.ch> HB9HUH/K6CFT
# SW released under the MIT license

import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--print', action='store_true')

args = parser.parse_args(sys.argv[1:])
print(args)

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
fig,axes = plt.subplots(4,9, figsize=(16,7), dpi=72)
fig.suptitle("\nGallery of Colored Noise Types: Introducing the Complementary Hues 'reddish' and 'blueish' (the sum of their power is constant over all frequencies)", fontweight='bold', fontsize='large')

x = 1 + np.arange(MAXF)

# ---------------------------------------------------------------------------
ax = axes[0][0]
ax.plot(x, color(x, 2.))
ax.set_ylabel("weight | linear")

ax = axes[0][1]
ax.plot(x, color(x, 1.))

ax = axes[0][2]
ax.plot(x, color(x, 0.5))

ax = axes[0][3]
ax.plot(x, hue(x, 'reddish'))
ax.set_title('cos()', loc='left', fontsize='small')

ax = axes[0][4]
ax.plot(x, color(x, 0.))

ax = axes[0][5]
ax.plot(x, hue(x, 'blueish'))
ax.set_title('sin()', loc='left', fontsize='small')

ax = axes[0][6]
ax.plot(x, color(x, -0.5))

ax = axes[0][7]
ax.plot(x, color(x, -1.))

ax = axes[0][8]
ax.plot(x, color(x, -2.))


# ---------------------------------------------------------------------------
ax = axes[1][0]
ax.plot(x, np.pow(color(x, 2.),2))
ax.set_ylabel("power | linear", fontweight='bold')

ax = axes[1][1]
ax.plot(x, np.pow(color(x, 1.),2))

ax = axes[1][2]
ax.plot(x, np.pow(color(x, 0.5),2))

ax = axes[1][3]
ax.plot(x, np.pow(hue(x, 'reddish'),2), 'orange')
for s in ax.spines.__dict__['_dict'].values():
    s.set_linewidth(2)

ax = axes[1][4]
ax.plot(x, np.pow(color(x, 0.),2))

ax = axes[1][5]
ax.plot(x, np.pow(hue(x, 'blueish'),2), 'orange')
for s in ax.spines.__dict__['_dict'].values():
    s.set_linewidth(2)

ax = axes[1][6]
ax.plot(x, np.pow(color(x, -0.5),2))

ax = axes[1][7]
ax.plot(x, np.pow(color(x, -1.),2))

ax = axes[1][8]
ax.plot(x, np.pow(color(x, -2.),2))


# ---------------------------------------------------------------------------
R1=[0.0000001,1]
R2=[0.9,19000000]
R3=[0.1,10]

ax = axes[2][0]
ax.loglog(x, 1e7 * np.pow(color(x, 2.),2))
ax.set_ylabel("power | loglog")
ax.set_ylim(R2)

ax = axes[2][1]
ax.loglog(x, 1e7 * np.pow(color(x, 1.),2), 'r')
ax.set_ylim(R2)

ax = axes[2][2]
ax.loglog(x, np.pow(color(MAXF+1-x, -0.5),2))
ax.set_ylim(R2)

ax = axes[2][3]
ax.loglog(x, np.pow(hue(x, 'reddish'),2))
ax.set_ylim([0.0000001,10000000])

ax = axes[2][4]
ax.loglog(x, np.pow(color(x, 0.),2), 'r')
ax.set_ylim(R3)

ax = axes[2][5]
ax.loglog(x, np.pow(hue(x, 'blueish'),2))
ax.set_ylim([0.0000001,10000000])

ax = axes[2][6]
ax.loglog(x, np.pow(color(x, -0.5),2))
ax.set_ylim(R2)

ax = axes[2][7]
ax.loglog(x, np.pow(color(x, -1.),2), 'r')
ax.set_ylim(R2)

ax = axes[2][8]
ax.loglog(x, np.pow(color(x, -2.),2))
ax.set_ylim(R2)


# ---------------------------------------------------------------------------
ax = axes[3][0]
ax.semilogy(x, 1e7 * np.pow(color(x, 2.),2))
ax.set_ylabel("power | semilogy")
ax.set_xlabel("p=2.0\nbrown")
ax.set_ylim(R2)

ax = axes[3][1]
ax.semilogy(x, 1e7 * np.pow(color(x, 1.),2))
ax.set_xlabel("p=1.0\npink")
ax.set_ylim(R2)

ax = axes[3][2]
ax.semilogy(x, np.pow(color(MAXF+1-x, -0.5),2))
ax.set_xlabel("p=0.5\nlightpink")
ax.set_ylim(R2)

ax = axes[3][3]
ax.semilogy(x, np.pow(hue(x, 'reddish'),2))
ax.set_xlabel("\n'reddish'", fontweight='bold')
ax.set_ylim(R3)

ax = axes[3][4]
ax.semilogy(x, np.pow(color(x, 0.),2))
ax.set_xlabel("p=0.0\nwhite")

ax = axes[3][5]
ax.semilogy(x, np.pow(hue(x, 'blueish'),2))
ax.set_xlabel("\n'blueish'", fontweight='bold')
ax.set_ylim(R3)

ax = axes[3][6]
ax.semilogy(x, np.pow(color(x, -0.5),2))
ax.set_xlabel("p=-0.5\nlightblue")
ax.set_ylim(R2)

ax = axes[3][7]
ax.semilogy(x, np.pow(color(x, -1.),2))
ax.set_xlabel("p=-1.0\nblue")
ax.set_ylim(R2)

ax = axes[3][8]
ax.semilogy(x, np.pow(color(x, -2.),2))
ax.set_xlabel("p=-2.0\nviolet")
ax.set_ylim(R2)


# ---------------------------------------------------------------------------
plt.subplots_adjust(top=0.89, bottom=0.1, left=0.05, right=0.99)

if args.print:
    FNAME = 'nck-noise-gallery'
    for t in ['png', 'pdf']:
        plt.savefig(FNAME + '.' + t, format=t)
        print(f"--> {FNAME}.{t}")
else:
    plt.show()

# eof
