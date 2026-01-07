#!/usr/bin/env python3

# demo-nck.py
# simulation of Noise Color Keying (NCK)

# (C) Dec 2025 <christian.tschudin@unibas.ch>, HB9HUH/K6CFT
# SW released under the MIT license


import argparse
from datetime import datetime,UTC
from ft8_coding import FT8_CODING
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
parser.add_argument('-f', '--fs', type=int, default=12000,
                          help="sampling frequency in Hz")
parser.add_argument('-k', '--kr', type=int, default=20,
                          help="keying rate in Baud")
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
print(args)

nck = NCK(FS=args.fs, CF=args.centerfreq, BW=args.bw,
          KR=args.kr, USE_FFT=args.fft)

ft8 = FT8_CODING()

LEN = args.length
if LEN == 174: # apply FT8 encoding
    bits = [x for x in np.random.randint(2,size=77)]
    bits += ft8.crc14(bits)
    bits = ft8.ldpc_encode(bits)
    assert len(bits) == 174
else:
    bits = [ 0 if np.random.rand(1) < 0.5 else 1 for i in range(LEN) ]
audio = nck.modulate(bits)

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
w = 2 * args.bw // args.kr # samples per symbol
sent = ([0] * (2*args.bw//w)) + \
       [-1 if b else 1 for b in bits] + \
       ([0] * (2*args.bw//w + 2))
sent = np.array( [ sent[int((x+w/2)/w)] for x in range(len(r1)) ] )
ax.plot(duration * np.arange(len(sent))/len(sent), sent * 0.075, 'r')
ax.annotate('"0"', [duration-0.005,0.1-0.025], color='red')
ax.annotate('"1"', [duration-0.005,-0.1-0.025], color='red')
ax.annotate('red: modulation input', [0,np.min(r1)], color='red')


# --- print original bits and reception status to the terminal
bits = "".join([str(b) for b in bits])
print(f"sent= {bits}")
ax.set_title(bits)

msg = msg[:len(bits)]
llr = [ -4.5 if b else 4.5 for b in msg ]
msg = ''.join([str(b) for b in msg])
err, s = 0, ''
for i in range(len(bits)):
    if bits[i] == msg[i]:
        s += "\033[32m"
    else:
        s += "\033[31m"
        err += 1
    s += msg[i] + "\033[0m"

if err == 0:
    print(f"rcvd= {s} (no bit errors)")
else:
    print(f"rcvd= {s} ({err} bit errors, {int(100*err/len(bits) + 0.9)}%)")

if err > 0 and len(bits) == 174: # FT8 encoding
    x, corr = ft8.ldpc_decode(llr, 100)
    if x == 91:
        corr = [ str(x) for x in corr ][:91]
        bits = bits[:91]
        err2, s = 0, ''
        for i in range(len(bits)):
            if bits[i] == corr[i]:
                s += "\033[32m"
            else:
                s += "\033[31m"
                err2 ++ 1
            s += corr[i] + "\033[0m"
        if err2 == 0:
            print(f"\033[0;93mcorr\033[0m= {s} (no frame error)")
        else:
            print(f"corr= {s} ({err2} errors, {int(100*err2/len(bits) + 0.9)}%)")

now = f"{str(datetime.now(UTC))[:19]} UTC"
if args.snr == '':
    snr = ", no noise"
else:
    snr = f", SNR={args.snr}dB"
if err == 0:
    err = ", no errors"
else:
    err = f", {int(100*err/len(bits) + 0.9)}% errors"

fig.supylabel(f"Noise Color Keying (NCK): simulation {now}{snr}{err}", x=0.95)

if args.print:
    FNAME = 'demo-nck'
    for t in ['png', 'pdf']:
        plt.savefig(FNAME + '.' + t, format=t, dpi=150)
        print(f"--> {FNAME}.{t}")
else:
    plt.show()

# eof
