#!/usr/bin/env python3

# mk_nck-fer_plot.py
# renders the collected FER data (in one or more JSON files)

# (C) Jan 2026 <christian.tschudin@unibas.ch> HB9HUH/K6CFT
# SW released under the MIT license


import json
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------

def render(fname, ax):

    with open(fname, 'r') as f:
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
        ax.semilogy(snr_values[:len(data)], data, label=f'{kr} Baud')

    ax.set_ylim([1e-3,1.25])
    ax.set_xlabel("SNR/dB")

    ax.axhline(y=0.5, color='magenta', linestyle='dashdot')
    pos = sys.argv[1].rfind('.')
    FNAME = sys.argv[1] if pos < 0 else sys.argv[1][:pos]
    ax.text(10.3, 1.3, FNAME, va='top', rotation='vertical', fontsize=8)
    ax.text(10.3, 0.00105, simu["cfg"]["utc"], rotation='vertical', fontsize=8)

    ax.grid(True, which="both")
    ax.legend(loc='lower right')

    bw = simu['cfg']['bw']
    fec = simu['cfg']['ecc']
    fec = "no FEC" if fec == None else f"FEC={fec}"
    counts = f"{simu['cfg']['dlength']}+{simu['cfg']['olength']} bits"

    ax.set_title(f"BW={bw}Hz / {fec} / {counts}",
                 fontsize=11)

# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    chart_cnt = len(sys.argv) - 1

    fig, axes = plt.subplots(nrows=1, ncols=chart_cnt,
                             sharey=True,
                             figsize=(5.5*chart_cnt,4)) #, dpi=90)

    for i,fn in enumerate(sys.argv[1:]):
        if chart_cnt == 1:
            ax = axes
        else:
            ax = axes[i]
        if i == 0:
            ax.set_ylabel("Frame Error Rate (FER)")
        render(fn, ax)

    if chart_cnt > 1:
        plt.suptitle(f"Noise Color Keying: Comparison of" + \
                     " Different Keying Rates and FEC Schemas",
                     fontsize=14)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85, bottom=0.13, left=0.06, right=0.975)
    else:
        plt.suptitle(f"Noise Color Keying: Comparison of" + \
                     " Different Keying Rates",
                     fontsize=12)
        plt.subplots_adjust(top=0.85, bottom=0.13, left=0.13, right=0.95)

    if chart_cnt == 1:
        FNAME = sys.argv[1]
        pos = FNAME.rfind('.')
        if pos > 0:
            FNAME = FNAME[:pos]
    else:
        FNAME = 'fer_plots'
    for t in ['png', 'pdf']:
        fig.savefig(FNAME + '.' + t, format=t, dpi=100)
        print(f"--> {FNAME}.{t}")

    plt.show()

# eof
