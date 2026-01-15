#!/usr/bin/env python3

# mk_nck-fer_simulation.py

# simulates NCK for various keying rate and SNR settings,
# persists the gathered FER values in a JSON file. That
# JSON file can be rendered with ./mk_nck-fer_plot.py

# (C) Jan 2026 <christian.tschudin@unibas.ch>, HB9HUH/K6CFT
# SW released under the MIT license


# example usage: start the first time with all parameters, interrupt
# and restart at any time and just pass the JSON file name
# 
# % ./mk_nck-fer_simulation.py -p simu-500-golay24-92.json -b 500 -e golay24 -l 91 -f 1000 -k 70,50,35,20
#      ^C
# % ./mk_nck-fer_simulation.py -p simu-500-golay24-92.json
# skipping kr=70.0 snr=-2.0 ecc=golay24
# ...

import argparse
from datetime import datetime,UTC
from ft8_coding import FT8_CODING
from golay24 import golay_encode, golay_decode
from hamming84 import h84_encode, h84_decode, h84_data_from_code
import json
import ncklib
import numpy as np
import os
import sys

# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--bw', type=int, default=500,
                          help="signal bandwidth in Hz (channel BW is 2700Hz)")
parser.add_argument('-c', '--centerfreq', type=int, default=0)
parser.add_argument('-e', '--ecc', choices=['FT8', 'golay24', 'hamming84'],
                          default=None,
                          help="use error correcting coding. Default=None")
parser.add_argument('-f', '--fs', type=int, default=6000,
                          help="sampling frequency in Hz")
parser.add_argument('-k', '--krl', type=str, default='20',
                          help="comma-separated list of keying rates in Baud")
parser.add_argument('-l', '--length', type=int, default='48',
                          help="payload len in bits. Default=48." + \
                               " Is adusted depending on -ecc")
parser.add_argument('-p', '--persist', type=str, metavar='FILENAME',
                          help="persist values, or resume if exists")
parser.add_argument('-r', '--rounds', type=int, default=3000,
                          help="number of simulation rounds. Use 3000 or more")
parser.add_argument('-t', '--fft', action='store_true',
                          help="use FFT instead of our LPF,HPF")

args = parser.parse_args(sys.argv[1:])
args.krl = [ float(x) for x in args.krl.split(',') ]

if args.persist != None:
    if os.path.isfile(args.persist):
        with open(args.persist, 'r') as f:
            simu = json.load(f)
        args.bw       = simu['cfg']['bw']
        args.ecc      = simu['cfg']['ecc']
        args.fs       = simu['cfg']['fs']
        args.krl      = simu['cfg']['krl']
        args.length   = simu['cfg']['dlength'] # data length
        args.overhead = simu['cfg']['olength'] # overhead length
        args.rounds   = simu['cfg']['rounds']
    else:
        if args.ecc == 'FT8':
            args.length = 91
            args.overhead = 174 - 91
        elif args.ecc == 'golay24':
            args.length = 12 * ((args.length + 11) // 12)
            args.overhead = args.length
        elif args.ecc == 'hamming84':
            args.length = 4 * ((args.length + 3) // 4)
            args.overhead = args.length
        else:
            args.overhead = 0
        print(args)

        simu = {}
        simu['cfg'] = {
            'bw'     : args.bw,
            'ecc'    : args.ecc,
            'fs'     : args.fs,
            'krl'    : args.krl,
            'dlength': args.length,
            'olength': args.overhead,
            'rounds' : args.rounds,
            'utc'    : str(datetime.now(UTC))[:19]
        }
        simu['data'] = {}
        with open(args.persist, 'w') as f:
            json.dump(simu, f)


ft8 = FT8_CODING()

def one_round():
    nck = ncklib.NCK(FS=args.fs, CF=args.centerfreq, BW=args.bw,
                     KR=args.kr, USE_FFT=args.fft)

    if args.ecc == 'FT8':
        data = [x for x in np.random.randint(2,size=77)]
        data += ft8.crc14(data)
        bits = ft8.ldpc_encode(data)
        assert len(bits) == 174
    else:
        data = np.random.randint(2, size=args.length)
        if args.ecc == 'golay24':
            bits = sum([golay_encode(data[12*i:12*i+12]) \
                        for i in range(len(data)//12)], [])
        elif args.ecc == 'hamming84':
            bits = sum([h84_encode(data[4*i:4*i+4]) \
                        for i in range(len(data)//4)], [])
        else:
            bits = [ x for x in data ]

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
    bband, r1, msg, _ = nck.demodulate(rcvd, msgstart=nck.FS)

    msg = msg[:len(bits)]
    msgstr = ''.join([str(b) for b in msg])
    err = 0
    for i in range(len(bits)):
        if bits[i] != msgstr[i]:
            err += 1

    frame_err = 0
    if args.ecc == 'FT8':
        llr = [ -4.5 if b else 4.5 for b in msg ]
        x, corr = ft8.ldpc_decode(llr, 100)
        if x != 91:
            frame_err += 1
    elif args.ecc == 'golay24':
        corr = []
        for i in range(len(msg)//24):
            corr += golay_decode(msg[i*24:i*24+24])
        for i in range(len(data)):
            if data[i] != corr[i]:
                frame_err += 1
                break
    elif args.ecc == 'hamming84':
        corr = []
        for i in range(len(msg)//8):
            ok, b4 = h84_decode(msg[i*8:i*8+8])
            corr += b4
        for i in range(len(data)):
            if data[i] != corr[i]:
                frame_err += 1
                break
    else:
        frame_err = 1 if err > 0 else 0

    return err, frame_err
    # end of one_round()

for kr in args.krl:
    args.kr = kr
    lst = [ str(v/2 - 2) for v in range(24) ]
    lowest_fer = 1.0
    for snr in lst:
        if args.persist != None:
            with open(args.persist, 'r') as f:
                simu = json.load(f)
            if not str(kr) in simu['data']:
                simu['data'][str(kr)] = {}
            if str(snr) in simu['data'][str(kr)]:
                print(f"skipping kr={kr} snr={snr} ecc={args.ecc}")
                fer = simu['data'][str(kr)][str(snr)]
                fer = float(fer[fer.rfind('=')+1:])
                if fer < lowest_fer:
                    lowest_fer = fer
                continue
            if lowest_fer <= 1e-3:
                break

        args.snr = snr
        bit_err_sum = 0
        frame_err_sum = 0

        for i in range(args.rounds):
            berr, ferr = one_round()
            bit_err_sum += berr
            frame_err_sum += ferr
            fer_rate = (frame_err_sum/(i+1))
            '''
            if frame_err_sum >= 60:
                print(f"kr={args.kr} snr={args.snr} rounds={i+1} berrs={bit_err_sum} ferrs={frame_err_sum} ber={'%e' % (bit_err_sum / (91*(i+1)))} fer={'%e' % fer_rate}")
                break
        else:
            print(f"snr={args.snr} rounds={args.rounds} berrs={bit_err_sum} ferrs={frame_err_sum} ber<{'%e' % (bit_err_sum / (91*(args.rounds+1)))} fer<{'%e' % (frame_err_sum/args.rounds)}")
            '''
            if frame_err_sum >= 60:
                line = f"kr={args.kr} snr={args.snr} rounds={i+1} fer={'%e' % fer_rate}"
                print(line)
                break
        else:
            line = f"kr={args.kr} snr={args.snr} rounds={args.rounds} fer={'%e' % (frame_err_sum/args.rounds)}"
            print(line)

        if args.persist != None:
            simu['data'][str(kr)][str(snr)] =  line
            simu["cfg"]["utc"] = str(datetime.now(UTC))[:19]
            with open(args.persist, 'w') as f:
                json.dump(simu, f)

        if frame_err_sum/args.rounds <= 1e-3:
            break

# eof
