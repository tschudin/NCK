#!/usr/bin/env python3

# https://web.archive.org/web/20161203074728/http://jaganadhg.freeflux.net:80/blog/archive/2009/09/09/plotting-wave-form-and-spectrogram-the-pure-python-way.html

import sys
from pylab import *
from matplotlib import transforms
import wave

def show_wave_n_spec(fn):
    spf = wave.open(fn, 'rb')
    fs = spf.getframerate()
    sound_info = frombuffer(spf.readframes(-1), int16)
    spf.close()

    fig = plt.figure(figsize=(10, 8))
    row1, row2 = fig.subfigures(2, 1, height_ratios=[1, 1])
    # row1.title('Wave from and spectrogram of %s' % fn)

    ax = row1.subplots(1, 1)
    ax.plot(sound_info)
    ax.set_xlabel("t (samples)")
    ax.set_ylabel("amplitude (wav file)")

    plots = row2.subplots(1, 2, width_ratios=[6,1])
    row2.subplots_adjust(wspace=0.05)
    ax = plots[0]
    spectrogram = ax.specgram(sound_info,
                              sides = 'default',
                              Fs = fs,
                              NFFT = 1024, # 64
                              pad_to = 8192, # 4096,
                              noverlap = 20,
                              window = window_hanning,
                              vmin=13,vmax=50) # vmin=0.001)
    ax.set_ylim( [0,3000] )
    ax.set_xlabel("t (sec)")
    ax.set_ylabel("frequency (Hz)")

    sp, fr, t, img = spectrogram
    s = np.sum(sp, axis=1)
    s /= np.max(s)/120
    # s = (s * 120).astype(int)

    ax = plots[1]
    ax.stairs(s[:-1], fr, orientation='horizontal')
    ax.set_ylim( [0,3000] )
    ax.set_yticks([])
    ax.set_xticks([])

    show()

show_wave_n_spec('out.wav' if len(sys.argv) == 1 else sys.argv[1])

# eof
