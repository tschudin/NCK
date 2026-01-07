#!/usr/bin/env python3

# mk_nck-fer_simulation.py
# runs NCK experiments for various SNR and keyrate settings,
# reports the respective BER and FER values

# (C) Jan 2026 <christian.tschudin@unibas.ch>, HB9HUH/K6CFT
# SW released under the MIT license


# example usage:
# 
# % ./mk_nck-fer_simulation.py -k 300
# Namespace(bw=2500, centerfreq=0, fs=6000, kr=300, rounds=3000, fft=False, length=174)
# snr=-2.0 rounds=30 berrs=1457 ferrs=30 ber=5.336996e-01 fer=1.000000e+00
# snr=-1.5 rounds=30 berrs=1368 ferrs=30 ber=5.010989e-01 fer=1.000000e+00
# ...
#
# % ./mk_nck-fer_simulation.py -k 250
# Namespace(bw=2500, centerfreq=0, fs=6000, kr=250, rounds=3000, fft=False, length=174)
# snr=-2.0 rounds=30 berrs=1203 ferrs=30 ber=4.406593e-01 fer=1.000000e+00
# snr=-1.5 rounds=30 berrs=1064 ferrs=30 ber=3.897436e-01 fer=1.000000e+00
# ...
#
# % ./mk_nck-fer_simulation.py -k 200
# etc


import argparse
from ft8_coding import FT8_CODING
import ncklib
import numpy as np
import sys

# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--bw', type=int, default=2500,
                          help="signal bandwidth in Hz (channel BW is 2700Hz)")
parser.add_argument('-c', '--centerfreq', type=int, default=0)
parser.add_argument('-f', '--fs', type=int, default=6000,
                          help="sampling frequency in Hz")
parser.add_argument('-k', '--kr', type=int, default=20,
                          help="keying rate in Baud")
parser.add_argument('-r', '--rounds', type=int, default=3000,
                          help="number of simulation rounds. Use 3000 or more")
parser.add_argument('-t', '--fft', action='store_true',
                          help="use FFT instead of our LPF,HPF")

args = parser.parse_args(sys.argv[1:])
args.length = 174
print(args)

nck = ncklib.NCK(FS=args.fs, CF=args.centerfreq, BW=args.bw,
                 KR=args.kr, USE_FFT=args.fft)

ft8 = FT8_CODING()

LEN = args.length

def one_round():
    bits = [x for x in np.random.randint(2,size=77)]
    bits += ft8.crc14(bits)
    bits = ft8.ldpc_encode(bits)
    assert len(bits) == 174

    audio = nck.modulate(bits)

    audio /= np.max(np.abs(audio)) # normalize
    audioLen = len(audio)
    pwrS = np.sum(audio*audio)     # signal power

    # pad with 1sec silence on each side
    audio = np.hstack((np.zeros(nck.FS),audio,np.zeros(nck.FS)))

    noise = 2 * np.random.rand(len(audio)) - 1
    pwrN = np.sum(noise*noise)     # noise power for full channel BW (args.FS/2)
    # adjust for padding
    pwrN *= audioLen / len(audio)
    # adjust for signal bandwidth
    pwrN *= args.bw / (args.fs/2)
    # adjust to requested SNR level
    x = 10 * np.log10(pwrS/pwrN) - float(args.snr)
    noise *= np.sqrt(np.power(10, (x/10)))
    # add noise
    audio += noise

    audio /= np.max(np.abs(audio))
    bits = "".join([str(b) for b in bits]) # this is the msg we sent

    rcvd = np.array( [x for x in audio] ) # this is the audio we received
    duration = len(rcvd) / nck.FS
    # demodulate
    bband, r1, msg = nck.demodulate(rcvd, msgstart=nck.FS)

    msg = msg[:len(bits)]
    llr = [ -4.5 if b else 4.5 for b in msg ]
    msg = ''.join([str(b) for b in msg])
    err = 0
    for i in range(len(bits)):
        if bits[i] != msg[i]:
            err += 1

    frame_err = 0
    if len(bits) == 174:
        x, corr = ft8.ldpc_decode(llr, 100)
        if x != 91:
            frame_err += 1

    return err, frame_err


lst = [ str(v/2 - 2) for v in range(35) ]
for snr in lst:
    # print(snr)
    args.snr = snr
    bit_err_sum = 0
    frame_err_sum = 0

    for i in range(args.rounds):
        berr, ferr = one_round()
        bit_err_sum += berr
        frame_err_sum += ferr
        if frame_err_sum >= 30:
            print(f"snr={args.snr} rounds={i+1} berrs={bit_err_sum} ferrs={frame_err_sum} ber={'%e' % (bit_err_sum / (91*(i+1)))} fer={'%e' % (frame_err_sum/(i+1))}")
            break
    else:
        print(f"snr={args.snr} rounds={args.rounds} berrs={bit_err_sum} ferrs={frame_err_sum} ber<{'%e' % (bit_err_sum / (91*(args.rounds+1)))} fer<{'%e' % (frame_err_sum/(args.rounds+1))}")

# eof
