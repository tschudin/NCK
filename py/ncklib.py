#!/usr/bin/env python3

# ncklib.py
# noise color keying (NCK)

# (C) Dec 2025 - Jan 2026 <christian.tschudin@unibas.ch> HB9HUH/K6CFT
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

class INTERLEAVE:

    def __init__(self, N): # mostly copied from WSPR
        map = [None] * N
        unmap = [None] * N
        P = 0
        while P < N:
            for I in range(256):
                J = 0
                for _ in range(8):
                    J = (J << 1) | (I & 0x01)
                    I >>= 1
                if J < N:
                    # J = (J + 43) % N  # this is where we differ
                    map[P] = J
                    assert unmap[J] == None
                    unmap[J] = P
                    P += 1
        self.m = map
        self.rm = unmap

    def map(self, lst):
        return [ lst[self.m[i]] for i in range(len(lst)) ]

    def unmap(self, lst):
        return [ lst[self.rm[i]] for i in range(len(lst)) ]

    pass

# ---------------------------------------------------------------------------

class NCK:

    REDDISH = -1
    WHITE   =  0
    BLUEISH = +1

    def __init__(self, FS=12000, CF=1500, BW=1000, KR=75, M=2, USE_FFT=False):
        self.FS  = FS  # sampling freq, in Hz
        self.CF  = CF  # center freq, in Hz
        self.BW  = BW  # bandwidth, in Hz
        self.KR  = KR  # keying rate, in Baud
        self.M   = M   # number of levels per symbol
        self.USE_FFT = USE_FFT

    def _noise(self, hue):
        # generate "two symbols worth" of samples of "noise with a hue"
        #   where hue is a float in the interval [-1..+1]: -1 stands
        #   for 'reddish', 0 for 'white', and +1 for 'blueish'

        SPS2 = int(2 * 2 * self.BW / self.KR) # samples per symbol, doubled
        wn = 2 * np.random.rand(SPS2) - 1

        if hue == self.WHITE:
            return wn

        if self.USE_FFT:
            fd = np.fft.fft(wn)
            if hue == self.REDDISH:
                for i in range(len(fd)):
                    fd[i] *= np.abs(np.cos(np.pi * i / len(fd)))
            elif hue == self.BLUEISH:
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

            if hue == self.REDDISH:
                n = lpf(wn)
                # n = np.sin(2 * np.pi * SPS2/4/2 * np.arange(SPS2)/SPS2)
            elif hue == self.BLUEISH:
                n = hpf(wn)
                # n = np.sin(2 * np.pi * 3*SPS2/4/2 * np.arange(SPS2)/SPS2)
            else:
                rn = lpf(wn)
                bn = hpf(wn)
                f = abs(self.BLUEISH - hue) / 2
                n = np.sqrt(f) * rn + np.sqrt(1 - f) * bn # flat power spectr
        return n / np.max(np.abs(n))

    def modulate(self, symlst):
        # returns timedomain signal at selected FS, no padding
        # symlst: vector of index values in [0..M-1]
    
        w = int(2 * self.BW / self.KR)  # samples per symbol (when FS=2*BW)
        sym = self._noise(self.WHITE)[:w] # ramp up (raised cosine white noise)
        sym *= 0.5 * (1 - np.cos(np.pi * np.arange(w)/w))
        sig = sym
        if self.M == 2:
            for b in symlst:
                if self.CF != 0: # mixing will flip the frequenc range
                    b = 1 - b
                sym = self._noise([self.REDDISH,self.BLUEISH][b])
                sig = np.hstack((sig, sym[:w]))
        elif self.M == 3:
            for i,t in enumerate(symlst):
                sym = self._noise([self.REDDISH,0,self.BLUEISH][t])
                sig = np.hstack((sig, sym[:w]))
        elif self.M == 4:
            for i,s in enumerate(symlst):
                sym = self._noise([self.REDDISH, self.REDDISH/3,
                                   self.BLUEISH/3, self.BLUEISH][s])
                sig = np.hstack((sig, sym[:w]))
        else:
            assert False
        sym = self._noise(self.WHITE)[:w] # ramp down (raised cosine white n.)
        sym *= 0.5 * (np.cos(np.pi * np.arange(w)/w) - 1)
        sig = np.hstack((sig, sym))

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

    def demodulate(self, rcvd, msgstart=0, msglen=None):
        # returns a 3-tuple: (sig,r1,symlst,samplepos)
        # where sig     extracted baseband signal (time domain)
        #       r1      smoothed lag1 autocorrelate signal
        #       symlst  recovered list of symbols
        #       sp      positions where r1 was sampled

        invert = False
        if self.CF != 0: # mix down to baseband

            if self.CF >= self.BW:
                # filter out band of interest (avoid mixing noise where it
                # does not belong)
                sos = signal.butter(10, [self.CF - self.BW/2,
                                        self.CF + self.BW/2], 'pass',
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
                            fs=2*self.BW,  output='sos')
        r1 = signal.sosfiltfilt(sos, r1)
        if invert:
            r1 *= -1

        # translate given offset to new sampling frequency (2*BW)
        msgstart = int(2 * self.BW * msgstart / self.FS)

        # sample the r1 signal (=decode)
        if msglen == None:
            relevant = r1[msgstart:]
        else:
            relevant = r1[msgstart:msgstart+msglen]

        samplePos = [ w*i for i in range(len(relevant)//w) ]
        mi,mx = np.min(relevant), np.max(relevant)
        if self.M == 2:
            msg = [ 1 if relevant[p] < 0 else 0 for p in samplePos ]
        elif self.M == 3:
            mx = 0.9 * max(-mi, mx)
            mi = -mx
            d = (mx - mi) / 3
            msg = []
            for p in samplePos:
                if relevant[p] < mi+d:
                    msg.append(0)
                elif relevant[p] < mx-d:
                    msg.append(1)
                else:
                    msg.append(2)
        elif self.M == 4:
            mx = 0.9 * max(-mi, mx)
            mi = -mx
            d = (mx - mi) / 4
            msg = []
            for p in samplePos:
                if relevant[p] < mi+d:
                    msg.append(0)
                elif relevant[p] < 0:
                    msg.append(1)
                elif relevant[p] < mx-d:
                    msg.append(2)
                else:
                    msg.append(3)
        else:
            assert False

        return (rcvd, r1, msg, samplePos)

    pass

# eof
