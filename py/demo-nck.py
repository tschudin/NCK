#!/usr/bin/env python3

# demo-nck.py
# simulation of Noise Color Keying (NCK)

# (C) Dec 2025 <christian.tschudin@unibas.ch>, HB9HUH/K6CFT
# SW released under the MIT license


import argparse
from datetime import datetime,UTC
from ft8_coding import FT8_CODING
from golay24 import   golay_encode, golay_decode
from hamming84 import h84_encode, h84_decode, h84_data_from_code
import json
from ldpc96 import    l96_encode, l96_decode, l96_data_from_code
import matplotlib.pyplot as plt
from matplotlib import transforms
from ncklib import INTERLEAVE, NCK
import numpy as np
import pylab
import scipy.io.wavfile
import scipy.signal as signal
import sys


# ---------------------------------------------------------------------------
barker = {
     7: [0,0,0,1,1,0,1],
    11: [0,0,0,1,1,1,0,1,1,0,1],
    13: [0,0,0,0,0,1,1,0,0,1,0,1,0],

    # doubled syms (half the keying rate)
    14: [0,0,0,0,0,0,1,1,1,1,0,0,1,1], 
    22: [0,0,0,0,0,0,1,1,1,1,1,1,0,0,1,1,1,1,0,0,1,1], # <== 2nd best
    26: [0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,1,1,0,0,1,1,0,0], # <== good!

    # tripled syms (third of keying rate)
    21: [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,1,1,1], # <== too weak
    33: [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,0,0,0,1,1,1],
    39: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,
         0,0,0,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0],
    # 21: [0,0,0,1,1,0,1,0,0,0,1,1,0,1,0,0,0,1,1,0,1] # 3x7, not good

    # note: high w needs less redundancy (14 ok for w=50,
    # while 26, 33 or 39 required for w=15)
}

# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--bw', type=float, default=500,
                          help="signal bandwidth in Hz " + \
                               "(channel BW is 2700Hz). Default=500")
parser.add_argument('-B', '--barker', type=int, default=None,
                          choices=[7,11,13,14,22,26,21,33,39],
                          help="insert Barker seq. Default=None")
parser.add_argument('-c', '--centerfreq', type=int, default=1250,
                          help=" Default=1250")
parser.add_argument('-e', '--ecc', default=None,
                          choices=['ft8', 'golay24', 'hamming84', 'ldpc96'],
                          help="use error correcting coding. Default=None")
parser.add_argument('-f', '--fs', type=int, default=6000,
                          help="sampling frequency in Hz. Default=6000")
parser.add_argument('-i', '--interleave', action='store_true',
                          help="enable interleaving. Default: False")
parser.add_argument('-k', '--kr', type=float, default=20,
                          help="keying rate in Baud. Default=20. Can be a float and smaller than 1")
parser.add_argument('-l', '--length', type=int, default=48,
                          help="payload len in bits. Default=48." + \
                               " Is adusted depending on -ecc")
parser.add_argument('-M', '--arity', type=int, choices=[2,3,4], default=2,
                          help="# of distinguished noise levels, default=2")
parser.add_argument('-p', '--print', action='store_true',
                          help="generate PNG and PDF")
parser.add_argument('-s', '--snr', type=str, default=3, metavar='dB',
                          help="SNR is specific to the signal's bandwidth." +\
                               " Default=3. Use '' for no noise")
parser.add_argument('-t', '--fft', action='store_true',
                          help="use FFT instead of our LPF,HPF")
parser.add_argument('-w', '--wav', action='store_true',
                          help="generate audio file 'out.wav'")
parser.add_argument('-y', '--birdies', type=float, default=0)

args = parser.parse_args(sys.argv[1:])
if args.ecc == 'ft8': # FT8 payload
    args.length = 77
elif args.ecc == 'golay24':
    args.length = 12 * ((args.length + 11) // 12)
elif args.ecc == 'hamming84':
    args.length = 4 * ((args.length + 3) // 4)
elif args.ecc == 'ldpc96': # WSPR payload
    args.length = 50
args.w = int(2 * args.bw / args.kr) # width (r1 samples per symbol, >25 is good)

print(args)

if args.barker != None:
    assert args.arity == 2

nck = NCK(FS=args.fs, CF=args.centerfreq, BW=args.bw,
          KR=args.kr, M=args.arity, USE_FFT=args.fft)

data = [x for x in np.random.randint(2, size=args.length)]
if args.ecc == 'ft8':
    ft8 = FT8_CODING()
    data += ft8.crc14(data)
    bits = [x for x in ft8.ldpc_encode(data)]
    assert len(bits) == 174
elif args.ecc == 'golay24':
    bits = sum([golay_encode(data[12*i:12*i+12]) \
                for i in range(len(data)//12)], [])
elif args.ecc == 'hamming84':
    bits = sum([h84_encode(data[4*i:4*i+4]) \
                for i in range(len(data)//4)], [])
elif args.ecc == 'ldpc96': # N=96, K=50 (v=3,c=6 ldpc)
    bits = l96_encode(data)
    assert len(bits) == 96
else:
    bits = data
bits_orig = bits # after encoding but before interleaving

print(f"data= \033[0;93m{''.join([str(x) for x in data])}\033[0m",
      f"({len(data)} bits)")

if args.ecc != None:
    print(f"encd= {''.join([str(x) for x in bits])} ({len(bits)} bits)")

if args.interleave:
    interleave = INTERLEAVE(len(bits))
    bits = interleave.map(bits)
    print(f"itlv= {''.join([str(x) for x in bits])} ({len(bits)} bits)")

if args.arity == 2:
    if args.barker != None:
        i = len(bits)//2
        symlst = bits[:i] + barker[args.barker] + bits[i:]
    else:
        symlst = bits
elif args.arity == 3:
    # map 3 bits to 2 ternary digits
    assert len(bits) % 3 == 0
    grps = [ bits[3*i:3*i+3] for i in range(len(bits)//3) ]
    vals = [ 4*g[0] + 2*g[1] + g[2] for g in grps ]
    symlst = sum([ [v//3, v%3] for v in vals ], [])
else:
    assert len(bits) % 2 == 0
    symlst = [ 2*bits[2*i] + bits[2*i+1] for i in range(len(bits)//2) ]

audio = nck.modulate(symlst)
xmit_time = len(audio)/nck.FS # includes the two ramp up/down symbols
sym_time = xmit_time / (1 + len(symlst) + 1)

''' # test for two parallel NCK signals (with -k 100 -b 1200 -c 1700)
nck2 = NCK(args=args, FS=args.fs, CF=600,
          BW=800, KR=args.kr)
bits2 = [ 0 if np.random.rand(1) < 0.5 else 1 for i in range(LEN) ]
audio2 = nck2.modulate(bits2)
print(audio2.shape)
audio2 /= np.max(np.abs(audio2)) # normalize

audio = audio + audio2[:len(audio)]
'''

audio /= np.max(np.abs(audio)) # normalize
audioLen = len(audio)
pwrS = np.sum(audio*audio)     # signal power

# pad with PADLEN sec silence before and after signal
PADLEN = 5 # in sec
audio = np.hstack((np.zeros(PADLEN*nck.FS),audio,np.zeros(PADLEN*nck.FS)))

if args.snr != '':
    noise = 2 * np.random.rand(len(audio)) - 1
    pwrN = np.sum(noise*noise)     # noise power for full channel BW
    # adjust for padding
    pwrN *= audioLen / len(audio)
    # adjust for signal bandwidth
    pwrN *= args.bw / (args.fs/2)
    # adjust to requested SNR level
    x = 10 * np.log10(pwrS/pwrN) - float(args.snr)
    noise *= np.sqrt(np.power(10, (x/10)))
    # add noise
    audio += noise

if args.birdies != 0:
    for i in range(3):
        audio += args.birdies * np.cos(2 * np.pi * \
                     (args.centerfreq + args.bw*i/2/4) * \
                                       (np.arange(len(audio))/nck.FS) )

audio /= np.max(np.abs(audio))
rcvd = np.array( [x for x in audio] ) # make a copy, this is what we receive

if args.wav:
    audio *= 14000
    audio = audio.astype('int16')
    FNAME = 'out.wav'
    scipy.io.wavfile.write(FNAME, nck.FS, audio)
    print(f"--> {FNAME}")
    j = {}
    for k in ['bw', 'barker', 'centerfreq', 'ecc', 'fs',
              'kr', 'length', 'arity', 'snr', 'w']:
        j[k] = args.__dict__[k]
    j['data'] = ''.join([str(b) for b in data])
    j['sent'] = ''.join([str(b) for b in symlst])
    j['duration'] = xmit_time
    j['utc'] = str(datetime.now(UTC))[:19]
    FNAME = 'out.json'
    with open(FNAME, 'w') as f:
        f.write(json.dumps(j, indent=2))
    print(f"--> {FNAME}")

symlststr = "".join([str(b) for b in symlst])
if args.barker != None:
    i = (len(symlststr) - args.barker) // 2
    s = symlststr[:i] + '.' + symlststr[i:i+args.barker]+ '.' + \
        symlststr[i+args.barker:]
else:
    s = symlststr
print(f"sent= {s} ({len(symlststr)} symbols, in {'%.1f'%xmit_time} sec)")

# --- create three plots
fig = plt.figure(figsize=(6.5, 8))
row1, row2, row3 = fig.subfigures(3, 1, height_ratios=[1, 1, 1])

# --- 1
plots = row1.subplots(1, 2, width_ratios=[7,1])
row1.subplots_adjust(wspace=0.05)
ax = plots[0]
spectrogram = ax.specgram(14000 * rcvd,
                          Fs = nck.FS,
                          NFFT = 512, # 1024, # 64
                          pad_to = 8192,
                          noverlap = 8,
                          window = pylab.window_hanning,
                          vmin=13, vmax=50)
ax.set_ylim( [0,2700] )
ax.set_xlabel("t (sec)")
ax.set_ylabel("frequency (Hz)")

sp, fr, t, img = spectrogram
s = np.sum(sp, axis=1)

ax = plots[1]
ax.stairs(s[:-1], fr, orientation='horizontal')
ax.set_ylim( [0,2700] )
ax.set_yticks([])
ax.set_xticks([])

duration = len(rcvd) / nck.FS # overall recording time, in sec
bband, r1, msg, pos = nck.demodulate(rcvd,
                         msgstart=PADLEN*nck.FS,
                         msglen = int((len(symlst)+2) * 2 * args.bw / args.kr))
# above msglen arg: is adjusted for the two ramp up/down symbols
r1 = r1[:-int(2 * args.bw / args.kr)] # cut last samples (1sym) with wild swings

# --- 2
ax = row2.subplots(1, 1)
ax.plot(duration * np.arange(len(bband))/len(bband), bband)
ax.set_ylabel("extracted subband")

duration -= 1/args.kr # wrt to r1

# --- 3
ax = row3.subplots(1, 1)
ax.plot(duration * np.arange(len(r1))/len(r1), r1, 'b')
ax.set_ylabel("lag 1 autocorrelation")

# generate curve of original sym values, to be overlayed on recovered r1 signal
if args.arity > 2:
    mi, mx = 0.9*np.min(r1), 0.9*np.max(r1)
    mx = 0.9 * max(-mi, mx)
    mi = -mx
    d = (mx - mi) / args.arity
    for i in range(args.arity+1):
        ax.axhline(y=mi + i*d, color='lightgreen', linestyle='dashed')
if args.arity == 2:
    # add ramp up/down symbols
    sent = [0] +  [-1 if b else 1 for b in symlst] + [0]
    ax.annotate('"0"', [duration-0.005,0.1-0.025], color='red')
    ax.annotate('"1"', [duration-0.005,-0.1-0.025], color='red')
elif args.arity == 3:
    sent = [0] + [s-1 for s in symlst] + [0]
    ax.annotate('"2"', [duration-0.005, 2*mx/3-0.025], color='lightgreen')
    ax.annotate('"1"', [duration-0.005,-0.025],        color='lightgreen')
    ax.annotate('"0"', [duration-0.005, 2*mi/3-0.025], color='lightgreen')
elif args.arity == 4:
    sent = [0] + [2*s/3-1 for s in symlst] + [0]
    ax.annotate('"3"', [duration-0.005,3*mx/4-0.025], color='lightgreen')
    ax.annotate('"2"', [duration-0.005,1*mx/4-0.025], color='lightgreen')
    ax.annotate('"1"', [duration-0.005,1*mi/4-0.025], color='lightgreen')
    ax.annotate('"0"', [duration-0.005,3*mi/4-0.025], color='lightgreen')

if args.barker != None:
    # search Barker sequence
    bas = []
    xcr = []
    for b in barker[args.barker]:
        bas += [-1 if b else 1] * args.w
    mx = 0
    mxpos = -1
    for i in range(len(r1) - len(bas)):
        s = 0
        for j in range(len(bas)):
            s += bas[j] * r1[i+j]
        xcr.append(s)
        if s > mx:
            mx = s
            mxpos = i
    xcr = np.array(xcr)

    barker_start = PADLEN + (1 + (len(symlst)-args.barker)//2 - 0.5) * sym_time

    duration2 = duration * (len(r1) - len(bas)) / len(r1)
    ax.plot(duration2 * np.arange(len(xcr))/len(xcr), 1. + xcr/500)
    ax.annotate(f'Barker "{args.barker}" crosscorrelation',
                [0,0.75], color='black')

    if mxpos >= 0:
        barker_detected = mxpos/args.w * sym_time
        print(f"Barker found at {np.round(barker_detected,5)}, ", end='')
        print(f"real pos: {np.round(barker_start,5)}:   ", end='')
        print(f"d={np.round(barker_start - barker_detected,5)} vs T={np.round(sym_time,5)} ", end='')
        err = np.round(100 * (barker_start - barker_detected) / sym_time)
        print(f"({abs(int(err))} %)")
        ax.axvline(barker_detected, linestyle='dashed', linewidth=0.5)

    # draw background reactangle where Barker sequence really is
    rect = plt.Rectangle((barker_start, -0.2), args.barker * sym_time, 0.4,
                         facecolor='#c0c0c0')
    ax.add_patch(rect)

# show sampling positions as thin green bars
pos = np.array(pos)[1:1+len(symlst)] + int(2 * args.bw * PADLEN)
ax.bar(duration * pos/len(r1), r1[pos], width=1/args.kr/5, color='lightgreen')

mod_input = np.array( [ sent[x//args.w] for x in range(len(sent)*args.w) ] )
mod_pos = np.arange(len(mod_input))/len(mod_input) * len(sent) - 0.5
ax.plot(PADLEN +  mod_pos * sym_time, mod_input * 0.075, 'r')
ax.axhline(0, color='black', linewidth=0.5)
ax.annotate('red: modulation input', [0,np.min(r1)], color='red')

ax.set_title(symlststr)

# --- print reception status to the terminal

def colordiff(ref, act, barker=None):
    err = 0
    s = ''
    if barker != None:
        i = (len(ref)-barker) // 2
        barker = [ i, i+barker ]
    else:
        barker = [ len(ref), -1 ]
    for i in range(len(ref)):
        if i in barker:
            s += '.'
        if ref[i] == act[i]:
            s += "\033[32m"
        else:
            s += "\033[31m"
            if i < barker[0] or i > barker[1]:
                err += 1
        s += str(act[i]) + "\033[0m"
    return err, s

msg = msg[1:1+len(symlst)]  # ignore rampup symbol
msgstr = ''.join([str(sym) for sym in msg])
err, s = colordiff(symlststr, msgstr, barker=args.barker)

print(f"rcvd= {s} ", end='')
if err == 0:
    print("(no symbol errors)")
else:
    print(f"({err} symbol errors, {int(100*err/len(msgstr) + 0.9)}%)")

if args.arity == 2:
    recovered = msg
else:
    if args.arity == 3:
        vals = [ min(7,3 * msg[2*i] + msg[2*i+1]) for i in range(len(msg)//2) ]
        recovered = sum([ [v//4, (v%4)//2, v%2] for v in vals], [])
        err, s = colordiff(bits, recovered)
        print(f"extr= {s}", end='')
    elif args.arity == 4:
        recovered = ''.join([bin(4+s)[-2:] for s in msg])
        recovered = [int(x) for x in recovered]
        err, s = colordiff(bits, recovered)
        print(f"extr= {s}", end='')
    if err > 0:
        print(f" ({err} bit errors, {int(100*err/len(recovered) + 0.9)}%)")
    else:
        print(f" (no bit errors)")

if args.barker != None:
    i = (len(recovered) - args.barker) // 2
    recovered_wo_barker = recovered[:i] + recovered[i+args.barker:]
else:
    recovered_wo_barker = recovered

if args.interleave:
    recovered_wo_barker = interleave.unmap(recovered_wo_barker)
    err, s = colordiff(bits_orig, recovered_wo_barker)
    print(f"vlti= {s} ", end='')
    if err == 0:
        print("(no symbol errors)")
    else:
        print(f"({err} symbol errors, {int(100*err/len(bits_orig) + 0.9)}%)")

if args.ecc == 'ft8':
    llr = [ -4.5 if b else 4.5 for b in recovered_wo_barker ]
    # llr = [ -8 * r1[p] for p in pos ]
    x, corr = ft8.ldpc_decode(llr, 100)
    err, s = colordiff(bits_orig, corr)
    print(f"corr= {s} ", end='')
    if err == 0:
        print("(no symbol errors)")
    else:
        print(f"({err} symbol errors, {int(100*err/len(corr) + 0.9)}%)")
    extr = ft8.ldpc_extract(corr)
    err, s = colordiff(data, extr)
elif args.ecc == 'golay24':
    extr = []
    for i in range(len(recovered_wo_barker)//24):
        extr += golay_decode(recovered_wo_barker[i*24:i*24+24])
    err, s = colordiff(data, extr)
elif args.ecc == 'hamming84':
    extr = []
    for i in range(len(recovered_wo_barker)//8):
        _, b4 = h84_decode(recovered_wo_barker[i*8:i*8+8])
        extr += b4
    err, s = colordiff(data, extr)
    extr = recovered
elif args.ecc == 'ldpc96':
    llr = [ -8 * r1[p] for p in pos ]
    if args.barker != None:
        i = (len(llr) - args.barker) // 2
        llr = llr[:i] + llr[i+args.barker:]
    if args.interleave:
        llr = interleave.unmap(llr)
    corr = l96_decode(llr)
    err, s = colordiff(bits_orig, corr)
    print(f"corr= {s} ", end='')
    if err == 0:
        print("(no symbol errors)")
    else:
        print(f"({err} symbol errors, {int(100*err/len(corr) + 0.9)}%)")
    extr = l96_data_from_code(corr)
    err, s = colordiff(data, extr)

print(f"data= {s} ", end='')
if err == 0:
    print(f"(frame OK)")
else:
    print(f"({err} bit errors, {int(100*err/len(data)+0.9)}%)")

now = f"{str(datetime.now(UTC))[:19]} UTC"
if args.snr == '':
    snr = ", no noise"
else:
    snr = f", SNR={args.snr}dB"
if err == 0:
    err = ", no errors"
else:
    err = f", {int(100*err/len(data) + 0.9)}% errors"

fig.supylabel(f"Noise Color Keying (NCK): simulation {now}{snr}{err}", x=0.95)

if args.print:
    FNAME = 'demo-nck'
    for t in ['png', 'pdf']:
        plt.savefig(FNAME + '.' + t, format=t, dpi=150)
        print(f"--> {FNAME}.{t}")
else:
    plt.show()

# eof
