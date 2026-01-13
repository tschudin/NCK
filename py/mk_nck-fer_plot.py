#!/usr/bin/env python3

# mk_nck-fer_plot.py
# renders the collected FER data (in a JSON file)

# (C) Jan 2026 <christian.tschudin@unibas.ch> HB9HUH/K6CFT
# SW released under the MIT license


from datetime import datetime,UTC
import json
import matplotlib.pyplot as plt
import sys

plt.figure(dpi=80)

with open(sys.argv[1], 'r') as f:
    simu = json.load(f)

snr_values = [ x/2-2. for x in range(31) ]

kr_list = sorted(simu['data'].keys(), key=lambda x:float(x), reverse=True)
for kr in kr_list:
    snr_list = sorted(simu['data'][kr].keys(), key=lambda x:float(x))
    data = []
    for snr in snr_list:
        line = simu['data'][kr][snr]
        v = [ float(x.split('=')[-1]) for x in line.split('\n') ]
        v = [ 1e-4 if x == 0 else x for x in v ]
        data.append(v)
    plt.semilogy(snr_values[:len(data)], data, label=f'{kr} Baud')

plt.ylim([1e-3,1.25])
plt.ylabel("Frame Error Rate (FER)")
plt.xlabel("SNR/dB")

plt.axhline(y=0.5, color='magenta', linestyle='dashdot')
pos = sys.argv[1].rfind('.')
FNAME = sys.argv[1] if pos < 0 else sys.argv[1][:pos]
plt.text(10.3, 1.3, FNAME, va='top', rotation='vertical')
plt.text(10.3, 0.00105, simu["cfg"]["utc"], rotation='vertical')

plt.grid(True, which="both")
plt.legend(loc='lower right')

bw = simu['cfg']['bw']
fec = simu['cfg']['ecc']
fec = "no FEC" if fec == None else f"FEC={fec}"
counts = f"{simu['cfg']['dlength']}+{simu['cfg']['olength']} bits"

plt.title(f"Comparison of Different Keying Rates for a {bw}Hz Signal\n" + \
          f"modulated with Noise Color Keying / {fec} / {counts}")

FNAME = sys.argv[1]
pos = sys.argv[1].rfind('.')
if pos > 0:
    FNAME = FNAME[:pos]
for t in ['png', 'pdf']:
    plt.savefig(FNAME + '.' + t, format=t) # , dpi=100)
    print(f"--> {FNAME}.{t}")

plt.show()

# eof
