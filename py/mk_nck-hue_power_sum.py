#!/usr/bin/env python3

# mk_nck-hue_power_sum.py
# creates reddish and blueish noise signals, stacks them up, and
# shows that they add up to white noise

# (C) Jan 2026 <christian.tschudin@unibas.ch> HB9HUH/K6CFT
# SW released under the MIT license

import argparse
import matplotlib.pyplot as plt
import numpy as np
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--fs', type=int, default=500)
parser.add_argument('-p', '--print', action='store_true')
parser.add_argument('-r', '--runs', type=int, default=5000)

args = parser.parse_args(sys.argv[1:])
print(args)
FS = args.fs

# ---------------------------------------------------------------------------

def lpf(v): # our low pass filter
    return np.array([ v[i] + v[i+1] for i in range(len(v)-1) ])

def hpf(v): # our high pass filter
    return np.array([ v[i] - v[i+1] for i in range(len(v)-1) ])

r_power = np.zeros(FS//2) # power spectrum for bandwidth FS//2
b_power = np.zeros(FS//2)

for i in range(args.runs):
    white = 2 * np.random.rand(FS) - 1 # one second of white noise

    # compute power using FFT, then accumulate
    # but randomly pick either of the two hues (we want to sum up
    # independent signals, check whether they result in white again)
    if np.random.rand(1) < 0.5:
        reddish  = lpf(white)               # lowpass filtering
        reddish /= np.max(np.abs(reddish))
        r_power += np.pow(np.abs(np.fft.rfft(reddish)),2)
    else:
        blueish  = hpf(white)               # highpass filtering
        blueish /= np.max(np.abs(blueish))
        b_power += np.pow(np.abs(np.fft.rfft(blueish)),2)

psum = np.average(r_power + b_power)

# ----------
fig,axes = plt.subplots(2,1, figsize=(6,8), dpi=100)
fig.suptitle(f"Power Distribution of {args.fs//2}Hz-Wide White Noise\n" + \
         f"after split into a Blueish and Reddish component ({args.runs} runs)")

x = np.arange(len(r_power))

# ----------
ax1 = axes[0]

ax1.plot(r_power + b_power, 'green', linewidth=1)
ax1.plot(r_power, '#ff8080', linewidth=2)
ax1.plot(b_power, '#8080ff', linewidth=3)

ax1.plot(psum * np.pow(np.cos(np.pi/2 * x/(args.fs/2)),2),
         'black', linestyle='dashed', linewidth=1)
ax1.plot(psum * np.pow(np.sin(np.pi/2 * x/(args.fs/2)),2),
         'black', linestyle='dashed', linewidth=1)

ax1.set_yticks([])
ax1.set_xlim([2, len(r_power)-1])
ax1.set_ylabel("Blue: blueish noise power. Red: reddish noise.\n" + \
               "Green: sum. Dashed: sin()^2. Note: Linear plot.")

# ----------
ax2 = axes[1]

ax2.loglog(r_power + b_power, 'green', linewidth=1)
ax2.loglog(r_power, '#ff8080', linewidth=2)
ax2.loglog(b_power, '#8080ff', linewidth=3)

ax2.loglog(psum * np.pow(np.cos(np.pi/2 * x/(args.fs/2)),2),
           'black', linestyle='dashed', linewidth=1)
ax2.loglog(psum * np.pow(np.sin(np.pi/2 * x/(args.fs/2)),2),
           'black', linestyle='dashed', linewidth=1)

ax2.set_ylabel("Blue: blueish noise power. Red: reddish noise.\n" + \
               "Green: sum. Dashed: sin()^2. Note: log-log plot.")

ax2.yaxis.tick_right()
ax2.set_xlim([2, len(r_power)-1])
ax2.set_xlabel("frequency (Hz)")


# ----------
plt.subplots_adjust(top=0.90, bottom=0.08, left=0.1, right=0.93)

if args.print:
    FNAME = 'nck-hue_power_sum'
    for t in ['png', 'pdf']:
        plt.savefig(FNAME + '.' + t, format=t)
        print(f"--> {FNAME}.{t}")
else:
    plt.show()

# eof
