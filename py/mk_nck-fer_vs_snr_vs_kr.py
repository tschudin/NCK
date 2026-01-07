#!/usr/bin/env python3

# mk_nck-fer_vs_snr_vs_kr.py
# renders the FER vs SNR graph
# (the simulation results have been copy-pasted by hand)

# (C) Jan 2026 <christian.tschudin@unibas.ch> HB9HUH/K6CFT
# SW released under the MIT license


import matplotlib.pyplot as plt

# Namespace(bw=2500, centerfreq=0, fs=12000, kr=300, length=174, rounds=2000, print=False, snr=6, fft=False, wav=False, birdies=0)
fer300a  = '''snr=-2.0 rounds=60 berrs=2944 ferrs=60 ber=5.391941e-01 fer=1.000000e+00
snr=-1.5 rounds=60 berrs=2803 ferrs=60 ber=5.133700e-01 fer=1.000000e+00
snr=-1.0 rounds=60 berrs=2585 ferrs=60 ber=4.734432e-01 fer=1.000000e+00
snr=-0.5 rounds=60 berrs=2427 ferrs=60 ber=4.445055e-01 fer=1.000000e+00
snr=0.0 rounds=60 berrs=2364 ferrs=60 ber=4.329670e-01 fer=1.000000e+00
snr=0.5 rounds=60 berrs=2209 ferrs=60 ber=4.045788e-01 fer=1.000000e+00
snr=1.0 rounds=61 berrs=2100 ferrs=60 ber=3.783102e-01 fer=9.836066e-01
snr=1.5 rounds=60 berrs=1901 ferrs=60 ber=3.481685e-01 fer=1.000000e+00
snr=2.0 rounds=60 berrs=1752 ferrs=60 ber=3.208791e-01 fer=1.000000e+00
snr=2.5 rounds=60 berrs=1664 ferrs=60 ber=3.047619e-01 fer=1.000000e+00
snr=3.0 rounds=60 berrs=1561 ferrs=60 ber=2.858974e-01 fer=1.000000e+00
snr=3.5 rounds=60 berrs=1430 ferrs=60 ber=2.619048e-01 fer=1.000000e+00
snr=4.0 rounds=62 berrs=1345 ferrs=60 ber=2.383906e-01 fer=9.677419e-01
snr=4.5 rounds=65 berrs=1377 ferrs=60 ber=2.327980e-01 fer=9.230769e-01
snr=5.0 rounds=66 berrs=1274 ferrs=60 ber=2.121212e-01 fer=9.090909e-01
snr=5.5 rounds=71 berrs=1266 ferrs=60 ber=1.959449e-01 fer=8.450704e-01
snr=6.0 rounds=69 berrs=1236 ferrs=60 ber=1.968466e-01 fer=8.695652e-01
snr=6.5 rounds=70 berrs=1161 ferrs=60 ber=1.822606e-01 fer=8.571429e-01
snr=7.0 rounds=92 berrs=1383 ferrs=60 ber=1.651935e-01 fer=6.521739e-01
snr=7.5 rounds=92 berrs=1353 ferrs=60 ber=1.616101e-01 fer=6.521739e-01
snr=8.0 rounds=99 berrs=1371 ferrs=60 ber=1.521812e-01 fer=6.060606e-01
snr=8.5 rounds=110 berrs=1481 ferrs=60 ber=1.479520e-01 fer=5.454545e-01
snr=9.0 rounds=150 berrs=1903 ferrs=60 ber=1.394139e-01 fer=4.000000e-01
snr=9.5 rounds=172 berrs=2118 ferrs=60 ber=1.353182e-01 fer=3.488372e-01
snr=10.0 rounds=167 berrs=1990 ferrs=60 ber=1.309469e-01 fer=3.592814e-01
snr=10.5 rounds=183 berrs=2145 ferrs=60 ber=1.288056e-01 fer=3.278689e-01
snr=11.0 rounds=225 berrs=2503 ferrs=60 ber=1.222466e-01 fer=2.666667e-01
snr=11.5 rounds=237 berrs=2602 ferrs=60 ber=1.206473e-01 fer=2.531646e-01
snr=12.0 rounds=248 berrs=2631 ferrs=60 ber=1.165810e-01 fer=2.419355e-01
snr=12.5 rounds=357 berrs=3736 ferrs=60 ber=1.149998e-01 fer=1.680672e-01
snr=13.0 rounds=308 berrs=3141 ferrs=60 ber=1.120665e-01 fer=1.948052e-01'''

# Namespace(bw=2500, centerfreq=0, fs=6000, kr=250, length=174, rounds=3000, print=False, snr=6, fft=False, wav=False, birdies=0)
fer250 = '''snr=-2.0 rounds=60 berrs=2443 ferrs=60 ber=4.474359e-01 fer=1.000000e+00
snr=-1.5 rounds=60 berrs=2264 ferrs=60 ber=4.146520e-01 fer=1.000000e+00
snr=-1.0 rounds=60 berrs=1983 ferrs=60 ber=3.631868e-01 fer=1.000000e+00
snr=-0.5 rounds=60 berrs=1829 ferrs=60 ber=3.349817e-01 fer=1.000000e+00
snr=0.0 rounds=30 berrs=854 ferrs=30 ber=3.128205e-01 fer=1.000000e+00
snr=0.5 rounds=30 berrs=789 ferrs=30 ber=2.890110e-01 fer=1.000000e+00
snr=1.0 rounds=30 berrs=681 ferrs=30 ber=2.494505e-01 fer=1.000000e+00
snr=1.5 rounds=31 berrs=632 ferrs=30 ber=2.240340e-01 fer=9.677419e-01
snr=2.0 rounds=31 berrs=597 ferrs=30 ber=2.116271e-01 fer=9.677419e-01
snr=2.5 rounds=37 berrs=627 ferrs=30 ber=1.862192e-01 fer=8.108108e-01
snr=3.0 rounds=43 berrs=626 ferrs=30 ber=1.599796e-01 fer=6.976744e-01
snr=3.5 rounds=49 berrs=630 ferrs=30 ber=1.412873e-01 fer=6.122449e-01
snr=4.0 rounds=86 berrs=1066 ferrs=30 ber=1.362126e-01 fer=3.488372e-01
snr=4.5 rounds=153 berrs=1627 ferrs=30 ber=1.168570e-01 fer=1.960784e-01
snr=5.0 rounds=242 berrs=2372 ferrs=30 ber=1.077105e-01 fer=1.239669e-01
snr=5.5 rounds=475 berrs=4073 ferrs=30 ber=9.422788e-02 fer=6.315789e-02
snr=6.0 rounds=600 berrs=4779 ferrs=30 ber=8.752747e-02 fer=5.000000e-02
snr=6.5 rounds=1010 berrs=7134 ferrs=30 ber=7.761941e-02 fer=2.970297e-02
snr=7.0 rounds=1622 berrs=10706 ferrs=30 ber=7.253289e-02 fer=1.849568e-02
snr=7.5 rounds=3000 berrs=17614 ferrs=24 ber<6.452015e-02 fer=8.000000e-03
snr=8.0 rounds=3000 berrs=16237 ferrs=15 ber<5.947619e-02 fer=5.000000e-03
snr=8.5 rounds=3000 berrs=15045 ferrs=10 ber<5.510989e-02 fer=3.333333e-03'''

fer200 = '''snr=-2.0 rounds=60 berrs=2056 ferrs=60 ber=3.765568e-01 fer=1.000000e+00
snr=-1.5 rounds=60 berrs=1851 ferrs=60 ber=3.390110e-01 fer=1.000000e+00
snr=-1.0 rounds=60 berrs=1684 ferrs=60 ber=3.084249e-01 fer=1.000000e+00
snr=-0.5 rounds=60 berrs=1480 ferrs=60 ber=2.710623e-01 fer=1.000000e+00
snr=0.0 rounds=33 berrs=737 ferrs=30 ber=2.454212e-01 fer=9.090909e-01
snr=0.5 rounds=31 berrs=573 ferrs=30 ber=2.031195e-01 fer=9.677419e-01
snr=1.0 rounds=42 berrs=707 ferrs=30 ber=1.849817e-01 fer=7.142857e-01
snr=1.5 rounds=52 berrs=779 ferrs=30 ber=1.646238e-01 fer=5.769231e-01
snr=2.0 rounds=62 berrs=798 ferrs=30 ber=1.414392e-01 fer=4.838710e-01
snr=2.5 rounds=128 berrs=1425 ferrs=30 ber=1.223386e-01 fer=2.343750e-01
snr=3.0 rounds=140 berrs=1363 ferrs=30 ber=1.069859e-01 fer=2.142857e-01
snr=3.5 rounds=366 berrs=3146 ferrs=30 ber=9.445746e-02 fer=8.196721e-02
snr=4.0 rounds=887 berrs=6529 ferrs=30 ber=8.088755e-02 fer=3.382187e-02
snr=4.5 rounds=2000 berrs=12725 ferrs=30 ber=6.991758e-02 fer=1.500000e-02
snr=5.0 rounds=3000 berrs=16685 ferrs=21 ber<6.111722e-02 fer=7.000000e-03
snr=5.5 rounds=3000 berrs=14456 ferrs=12 ber<5.295238e-02 fer=4.000000e-03
snr=6.0 rounds=3000 berrs=12646 ferrs=2 ber<4.632234e-02 fer=6.666667e-04'''

fer150 = '''snr=-2.0 rounds=60 berrs=1729 ferrs=60 ber=3.166667e-01 fer=1.000000e+00
snr=-1.5 rounds=60 berrs=1576 ferrs=60 ber=2.886447e-01 fer=1.000000e+00
snr=-1.0 rounds=60 berrs=1321 ferrs=60 ber=2.419414e-01 fer=1.000000e+00
snr=-0.5 rounds=65 berrs=1252 ferrs=60 ber=2.116653e-01 fer=9.230769e-01
snr=0.0 rounds=41 berrs=691 ferrs=30 ber=1.852050e-01 fer=7.317073e-01
snr=0.5 rounds=39 berrs=597 ferrs=30 ber=1.682164e-01 fer=7.692308e-01
snr=1.0 rounds=81 berrs=1013 ferrs=30 ber=1.374305e-01 fer=3.703704e-01
snr=1.5 rounds=139 berrs=1451 ferrs=30 ber=1.147126e-01 fer=2.158273e-01
snr=2.0 rounds=218 berrs=1997 ferrs=30 ber=1.006654e-01 fer=1.376147e-01
snr=2.5 rounds=668 berrs=5237 ferrs=30 ber=8.615187e-02 fer=4.491018e-02
snr=3.0 rounds=1826 berrs=12126 ferrs=30 ber=7.297522e-02 fer=1.642935e-02
snr=3.5 rounds=3000 berrs=16896 ferrs=21 ber<6.189011e-02 fer=7.000000e-03
snr=4.0 rounds=3000 berrs=14150 ferrs=9 ber<5.183150e-02 fer=3.000000e-03
snr=4.5 rounds=3000 berrs=12315 ferrs=1 ber<4.510989e-02 fer=3.333333e-04'''

fer125 = '''snr=-2.0 rounds=63 berrs=1384 ferrs=60 ber=2.414094e-01 fer=9.523810e-01
snr=-1.5 rounds=69 berrs=1295 ferrs=60 ber=2.062430e-01 fer=8.695652e-01
snr=-1.0 rounds=100 berrs=1528 ferrs=60 ber=1.679121e-01 fer=6.000000e-01
snr=-0.5 rounds=132 berrs=1718 ferrs=60 ber=1.430236e-01 fer=4.545455e-01
snr=0.0 rounds=118 berrs=1259 ferrs=30 ber=1.172472e-01 fer=2.542373e-01
snr=0.5 rounds=355 berrs=3210 ferrs=30 ber=9.936542e-02 fer=8.450704e-02
snr=1.0 rounds=866 berrs=6407 ferrs=30 ber=8.130092e-02 fer=3.464203e-02
snr=1.5 rounds=2016 berrs=11980 ferrs=30 ber=6.530176e-02 fer=1.488095e-02
snr=2.0 rounds=3000 berrs=14428 ferrs=17 ber<5.284982e-02 fer=5.666667e-03
snr=2.5 rounds=3000 berrs=11266 ferrs=3 ber<4.126740e-02 fer=1.000000e-03'''

fer100 = '''snr=-2.0 rounds=79 berrs=1318 ferrs=60 ber=1.833357e-01 fer=7.594937e-01
snr=-1.5 rounds=97 berrs=1359 ferrs=60 ber=1.539594e-01 fer=6.185567e-01
snr=-1.0 rounds=241 berrs=2638 ferrs=60 ber=1.202864e-01 fer=2.489627e-01
snr=-0.5 rounds=510 berrs=4725 ferrs=60 ber=1.018100e-01 fer=1.176471e-01
snr=0.0 rounds=809 berrs=5911 ferrs=30 ber=8.029177e-02 fer=3.708282e-02
snr=0.5 rounds=3000 berrs=16849 ferrs=18 ber<6.171795e-02 fer=6.000000e-03
snr=1.0 rounds=3000 berrs=13251 ferrs=0 ber<4.853846e-02 fer=3.000000e-03
snr=1.5 rounds=3000 berrs=10106 ferrs=1 ber<3.701832e-02 fer=3.333333e-04'''

snr = [ x/2-2. for x in range(31) ]

fer300a = [ x.split('=')[-1] for x in fer300a.split('\n') ]
fer300 = [ float(x) for x in fer300a ]

fer250 = [ x.split('=')[-1] for x in fer250.split('\n') ]
fer250 = [ float(x) for x in fer250 ]

fer200 = [ x.split('=')[-1] for x in fer200.split('\n') ]
fer200 = [ float(x) for x in fer200 ]

fer150 = [ x.split('=')[-1] for x in fer150.split('\n') ]
fer150 = [ float(x) for x in fer150 ]

fer125 = [ x.split('=')[-1] for x in fer125.split('\n') ]
fer125 = [ float(x) for x in fer125 ]

fer100 = [ x.split('=')[-1] for x in fer100.split('\n') ]
fer100 = [ float(x) for x in fer100 ]

plt.semilogy(snr[:len(fer300)], fer300, 'b', label='300 Baud')
plt.semilogy(snr[:len(fer250)], fer250, 'r', label='250 Baud')
plt.semilogy(snr[:len(fer200)], fer200, 'g', label='200 Baud')
plt.semilogy(snr[:len(fer150)], fer150, 'black', label='150 Baud')
plt.semilogy(snr[:len(fer125)], fer125, 'magenta', label='125 Baud')
plt.semilogy(snr[:len(fer100)], fer100, 'orange', label='100 Baud')

plt.ylim([1e-3,1.25])
plt.ylabel("Frame Error Rate (FER)")
plt.xlabel("SNR/dB")

plt.grid(True, which="both")
plt.legend(loc='lower right')

plt.title("Comparison of Different Keying Rates for a 2500Hz Signal\n" + \
          "modulated with Noise Color Keying (NCK) and using FT8's LDPC FEC")

FNAME = 'nck-fer_vs_snr_vs_kr'
for t in ['png', 'pdf']:
    plt.savefig(FNAME + '.' + t, format=t, dpi=100)
    print(f"--> {FNAME}.{t}")

plt.show()

# eof
