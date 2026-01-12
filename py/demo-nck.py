#!/usr/bin/env python3

# demo-nck.py
# simulation of Noise Color Keying (NCK)

# (C) Dec 2025 <christian.tschudin@unibas.ch>, HB9HUH/K6CFT
# SW released under the MIT license


import argparse
from datetime import datetime,UTC
from ft8_coding import FT8_CODING
from hamming84 import h84_encode, h84_decode, h84_data_from_code
import matplotlib.pyplot as plt
from matplotlib import transforms
from ncklib import NCK
import numpy as np
import pylab
import scipy.io.wavfile
import scipy.signal as signal
import sys

# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--bw', type=int, default=500,
                          help="signal bandwidth in Hz (channel BW is 2700Hz)")
parser.add_argument('-c', '--centerfreq', type=int, default=1250)
parser.add_argument('-e', '--ecc', choices=['hamming84'], default=None,
                          help="error correcting code: hamming84")
parser.add_argument('-f', '--fs', type=int, default=12000,
                          help="sampling frequency in Hz")
parser.add_argument('-k', '--kr', type=float, default=20,
                          help="keying rate in Baud, can be smaller than 1")
parser.add_argument('-l', '--length', type=int, default=48,
                          help="msg len in bits, default is 48." + \
                               " Use 174 for FT8 comparison")
parser.add_argument('-p', '--print', action='store_true',
                          help="generate PNG and PDF")
parser.add_argument('-s', '--snr', type=str, default=3, metavar='dB',
                          help="SNR is specific to the signal's bandwidth." +\
                               " Use '' for no noise")
parser.add_argument('-t', '--fft', action='store_true',
                          help="use FFT instead of our LPF,HPF")
parser.add_argument('-w', '--wav', action='store_true',
                          help="generate audio file 'out.wav'")
parser.add_argument('-y', '--birdies', type=float, default=0)

args = parser.parse_args(sys.argv[1:])
if args.ecc == 'hamming84':
    args.length = 4 * ((args.length + 3) // 4)
print(args)

nck = NCK(FS=args.fs, CF=args.centerfreq, BW=args.bw,
          KR=args.kr, USE_FFT=args.fft)

ft8 = FT8_CODING()

LEN = args.length
if LEN == 174: # apply FT8 encoding
    data = [x for x in np.random.randint(2, size=77)]
    data += ft8.crc14(data)
    bits = ft8.ldpc_encode(data)
    assert len(bits) == 174
else:
    data = np.random.randint(2, size=LEN)
    if args.ecc == 'hamming84':
        bits = sum([h84_encode(data[4*i:4*i+4]) for i in range(len(data)//4)],
                   [])
    else:
        bits = [ x for x in data ]
        
audio = nck.modulate(bits)
print(f"transmission time = {'%.1f'%(len(audio)/nck.FS)} sec")

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

# pad with 1sec silence on each side
audio = np.hstack((np.zeros(nck.FS),audio,np.zeros(nck.FS)))

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

duration = len(rcvd) / nck.FS
bband, r1, msg = nck.demodulate(rcvd, msgstart=nck.FS)

# --- 2
ax = row2.subplots(1, 1)
ax.plot(duration * np.arange(len(bband))/len(bband), bband)
ax.set_ylabel("extracted subband")


# --- 3
ax = row3.subplots(1, 1)
ax.plot(duration * np.arange(len(r1))/len(r1), r1, 'b')
ax.set_ylabel("lag 1 autocorrelation")

# generate curve of original bit values, to be overlayed on recovered r1 signal
w = int(2 * args.bw / args.kr) # samples per symbol
sent = ([0] * (2*args.bw//w)) + \
       [-1 if b else 1 for b in bits] + \
       ([0] * (2*args.bw//w + 2))
sent = np.array( [ sent[int((x+w/2)/w)] for x in range(len(r1)) ] )
ax.plot(duration * np.arange(len(sent))/len(sent), sent * 0.075, 'r')
ax.annotate('"0"', [duration-0.005,0.1-0.025], color='red')
ax.annotate('"1"', [duration-0.005,-0.1-0.025], color='red')
ax.annotate('red: modulation input', [0,np.min(r1)], color='red')


# --- print original bits and reception status to the terminal
print(f"data= \033[0;33m{''.join([str(x) for x in data])}\033[0m",
      f"({len(data)} bits)")
bits = "".join([str(b) for b in bits])
print(f"sent= {bits} ({len(bits)} bits)")
ax.set_title(bits)

msg = msg[:len(bits)]
msgstr = ''.join([str(b) for b in msg])
err, s = 0, ''
for i in range(len(bits)):
    if bits[i] == msgstr[i]:
        s += "\033[32m"
    else:
        s += "\033[31m"
        err += 1
    s += msgstr[i] + "\033[0m"

if err == 0:
    print(f"rcvd= {s} (no bit errors)")
    extr = None
    if len(bits) == 174: # FT8 encoding
        extr = bits[:91]
    elif args.ecc == 'hamming84':
        extr = sum([h84_data_from_code(msg[i*8:i*8+8]) \
                                    for i in range(len(msg)//8) ], [])
    if extr != None:
        print(f"extr= \033[32m{''.join([str(x) for x in extr])}\033[0m",
              "(no frame error)")
else:
    print(f"rcvd= {s} ({err} bit errors, {int(100*err/len(msgstr) + 0.9)}%)")
    if len(bits) == 174: # FT8 encoding
        llr = [ -4.5 if b else 4.5 for b in msg ]
        x, corr = ft8.ldpc_decode(llr, 100)
        if x == 91:
            extr = corr[:91]
        else:
            extr = msg[:91]
        err2, s = 0, ''
        for i in range(len(data)):
            if data[i] == extr[i]:
                s += "\033[32m"
            else:
                s += "\033[31m"
                err2 += 1
            s += str(extr[i]) + "\033[0m"
        if err2 == 0:
            print(f"\033[0;93mcorr\033[0m= {s} (no frame error)")
        else:
            print(f"extr= {s} ({err2} bit errors, {int(100*err2/len(data)+0.9)}%)")
    elif args.ecc == 'hamming84':
        corr = []
        for i in range(len(msg)//8):
            ok, b4 = h84_decode(msg[i*8:i*8+8])
            corr += b4
        err2, s = 0, ''
        for i in range(len(data)):
            if data[i] == corr[i]:
                s += "\033[32m"
            else:
                s += "\033[31m"
                err2 += 1
            s += str(corr[i]) + "\033[0m"
        print(f"\033[0;93mcorr\033[0m= {s} ", end='')
        if err2 == 0:
            print(f"(no frame error)")
        else:
            print(f"({err2} errors, {int(100*err2/len(data)+0.9)}%)")

now = f"{str(datetime.now(UTC))[:19]} UTC"
if args.snr == '':
    snr = ", no noise"
else:
    snr = f", SNR={args.snr}dB"
if err == 0:
    err = ", no errors"
else:
    err = f", {int(100*err/len(msg) + 0.9)}% errors"

fig.supylabel(f"Noise Color Keying (NCK): simulation {now}{snr}{err}", x=0.95)

if args.print:
    FNAME = 'demo-nck'
    for t in ['png', 'pdf']:
        plt.savefig(FNAME + '.' + t, format=t, dpi=150)
        print(f"--> {FNAME}.{t}")
else:
    plt.show()

# eof
