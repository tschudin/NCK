#!/usr/bin/env python3

# ncklib.py
# noise color keying (NCK)

# (C) Dec 2025 <christian.tschudin@unibas.ch> HB9HUH/K6CFT
# SW released under the MIT license

import numpy as np
import scipy.signal as signal

# ---------------------------------------------------------------------------
# fast non-class-based implementation of the lag1 autocorrelation computation
# note: this is not thread-safe because of global variables

def lag1autocorr_init(n):
    global old_x, old_avg, old_x_sum, x_min_avg, r1_N

    r1_N = n
    old_x     = [0] * r1_N
    x_min_avg = [0] * r1_N
    old_avg   = 0
    old_x_sum = 0

def lag1autocorr(v): # runs in 20% of the time of the naive impl.
    global old_x, old_avg, old_x_sum, x_min_avg, r1_N

    old_x_sum += v - old_x[0]
    avg = old_x_sum / r1_N

    old_x = old_x[1:] + [v]
    x_min_avg = x_min_avg[1:] + [v - old_avg]

    d = avg - old_avg
    old_avg = avg

    s2 = 0.
    for t in range(len(x_min_avg)):
        v = x_min_avg[t] - d
        x_min_avg[t] = v
        s2 += v * v

    s1 = 0.
    for t in range(len(x_min_avg)-1):
        s1 += x_min_avg[t] * x_min_avg[t+1]

    return s1 / s2

def lag1autocorr_naive(v): # close to the r1 formula, but slow
    global old_x, r1_N

    old_x = old_x[1:] + [v]
    avg = np.mean(old_x)
    s1 = 0.
    for t in range(r1_N-1):
        s1 += (old_x[t] - avg) * (old_x[t+1] - avg)
    s2 = 0.
    for t in range(r1_N):
        s2 += np.pow((old_x[t] - avg), 2)
    return s1 / s2

# ---------------------------------------------------------------------------

class NCK:

    def __init__(self, FS=12000, CF=1500, BW=1000, KR=75, USE_FFT=False):
        self.FS  = FS
        self.CF  = CF
        self.BW  = BW
        self.KR  = KR
        self.USE_FFT = USE_FFT

    def _noise(self, hue):
        # generate "two symbols worth" of samples of "noise with a hue"
        #   where hue is in ['reddish', 'white', 'blueish']

        SPS2 = int(2 * 2 * self.BW / self.KR) # samples per symbol, doubled
        wn = 2 * np.random.rand(SPS2) - 1

        if hue == 'white':
            return wn

        if self.USE_FFT:
            fd = np.fft.fft(wn)
            if hue == 'reddish':
                for i in range(len(fd)):
                    fd[i] *= np.abs(np.cos(np.pi * i / len(fd)))
            elif hue == 'blueish':
                for i in range(len(fd)):
                    fd[i] *= np.sin(np.pi * i / len(fd))
            else:
                assert False
            n = np.fft.ifft(fd).real
        else:
            def lpf(v): # our low pass filter
                return np.array([ v[i] + v[i+1] for i in range(len(v)-1) ])

            def hpf(v): # our high pass filter
                return np.array([ v[i] - v[i+1] for i in range(len(v)-1) ])

            if hue == 'reddish':
                n = lpf(wn)
            elif hue == 'blueish':
                n = hpf(wn)
            else:
                assert False
        return n / np.max(np.abs(n))

    def modulate(self, bits):
        # returns timedomain signal at selected FS, no padding
    
        w = int(2 * self.BW / self.KR)  # samples per symbol (when FS=2*BW)
        sig = np.zeros(0)
        for b in bits:
            if self.CF != 0: # mixing will flip the frequenc range
                b = 1 - b
            sym = self._noise(['reddish','blueish'][b])
            sig = np.hstack((sig, sym[:w]))

        if self.CF != 0:
            if self.CF >= self.BW:
                # increase FS (to cut off at CF+BW/2, eliminates mirror image)
                tmp_fs = self.CF + self.BW//2
                sig = signal.resample(sig, len(sig) * tmp_fs // self.BW)
                # transpose baseband to CF
                sig *= np.cos(2 * np.pi * tmp_fs * \
                              (np.arange(len(sig)) / (2*tmp_fs)))
            else:
                assert self.CF+3*self.BW/2 <= self.FS//2, \
                       "FS too small for mixing"
                # move signal up
                tmp_fs1 = self.FS // 2 - self.BW // 2
                sig = signal.resample(sig, len(sig) * tmp_fs1 // self.BW)
                sig *= np.cos(2 * np.pi * tmp_fs1 * \
                              (np.arange(len(sig)) / (2*tmp_fs1)))
                # move signal down
                tmp_fs2 = (tmp_fs1) - (self.CF + self.BW//2)
                sig *= np.cos(2 * np.pi * tmp_fs2 * \
                              (np.arange(len(sig)) / (2*tmp_fs1)))
                tmp_fs3 = self.CF + self.BW//2
                sig = signal.resample(sig, len(sig) * tmp_fs3 // tmp_fs1)
                tmp_fs = tmp_fs3
        else:
            tmp_fs = self.BW
        # upsample to final FS
        return signal.resample(sig, int(self.FS * len(sig) / (2*tmp_fs)))

    def demodulate(self, rcvd, msgstart=0):
        # returns a 3-tuple: (sig,r1,bits)
        # where sig  is the extracted baseband signal (time domain)
        #       r1   is the smoothed lag1 autocorrelate signal
        #       bits is the recovered bit vector

        invert = False
        if self.CF != 0: # mix down to baseband

            if self.CF >= self.BW:
                # filter out band of interest (avoid mixing noise where it
                # does not belong)
                sos = signal.butter(4, [self.CF - 3*self.BW/4,
                                        self.CF + 3*self.BW/4], 'pass',
                                    fs=self.FS,  output='sos')
                rcvd = signal.sosfiltfilt(sos, rcvd)
            else:
                print("warning: no bandpass filtering applied")

            if self.CF >= self.BW:
                s = np.cos(2 * np.pi * (self.CF-self.BW/2) * \
                           (np.arange(len(rcvd)) / self.FS))
                rcvd *= s
            else:
                s = np.cos(2 * np.pi * (self.FS/2) * \
                           (np.arange(len(rcvd)) / self.FS))
                rcvd *= s
                s = np.cos(2 * np.pi * (self.FS/2 - (self.CF + self.BW/2)) * \
                           (np.arange(len(rcvd)) / self.FS))
                rcvd *= s
                invert = True

        # apply lowpass filter at BW boundary
        rcvd = signal.resample(rcvd, int(2*self.BW * len(rcvd) / self.FS))
        rcvd /= np.max(np.abs(rcvd))
        
        w = int(2 * self.BW / self.KR) # samples per symbol
        rcvd = np.hstack( ([0.01]*w, rcvd, [0.01]*w) )
        lag1autocorr_init(w)

        r1 = [ lag1autocorr(x) for x in rcvd ][2*w:]
        # smooth according to sender's keying rate
        sos = signal.butter(2, self.KR, 'low',
                            fs=self.BW,  output='sos')
        r1 = signal.sosfiltfilt(sos, r1)
        if invert:
            r1 *= -1

        # translate given offset to new sampling frequency (2*BW)
        msgstart = int(2 * self.BW * msgstart / self.FS)
        _r1 = r1[msgstart:]

        # sample the r1 signal (=decode)
        msg = [ 1 if _r1[w*i] < 0 else 0 for i in range(len(_r1)//w) ]

        return (rcvd, r1, msg)

    pass

# eof
