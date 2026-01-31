# ldpc96.py

# code from pyldpc, https://github.com/hichamjanati/pyldpc

'''
BSD 3-Clause License

Copyright (c) 2019, Hicham Janati
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import numpy as np
import scipy

# ---------------------------------------------------------------------------

def _bitsandnodes(H):
    """Return bits and nodes of a parity-check matrix H."""
    if type(H) != scipy.sparse.csr_matrix:
        bits_indices, bits = np.where(H)
        nodes_indices, nodes = np.where(H.T)
    else:
        bits_indices, bits = scipy.sparse.find(H)[:2]
        nodes_indices, nodes = scipy.sparse.find(H.T)[:2]
    bits_histogram = np.bincount(bits_indices)
    nodes_histogram = np.bincount(nodes_indices)

    return bits_histogram, bits, nodes_histogram, nodes

def _incode(H, x):
    """Compute Binary Product of H and x."""
    return (H.dot(x) % 2 == 0).all()

def _gausselimination(A, b):
    """Solve linear system in Z/2Z via Gauss Gauss elimination."""
    A = A.copy()
    b = b.copy()
    n, k = len(A), len(A[0])

    for j in range(min(k, n)):
        listedepivots = [i for i in range(j, n) if A[i][j]]
        if len(listedepivots):
            pivot = min(listedepivots)
        else:
            continue
        if pivot != j:
            aux = (A[j]).copy()
            A[j] = A[pivot]
            A[pivot] = aux

            aux = b[j] # .copy()
            b[j] = b[pivot]
            b[pivot] = aux

        for i in range(j+1, n):
            if A[i][j]:
                A[i] = [ abs(A[i][c]-A[j][c]) for c in range(len(A[0])) ]
                b[i] = abs(b[i]-b[j])

    return A, b

# ---------------------------------------------------------------------------

def _logbp_numba(bits_hist, bits_values, nodes_hist, nodes_values, Lc, Lq, Lr,
                 n_iter):
    """Perform inner ext LogBP solver."""
    m, n, n_messages = Lr.shape
    # step 1 : Horizontal

    bits_counter = 0
    nodes_counter = 0
    for i in range(m):
        # ni = bits[i]
        ff = bits_hist[i]
        ni = bits_values[bits_counter: bits_counter + ff]
        bits_counter += ff
        for j in ni:
            nij = ni[:]

            X = np.ones(n_messages)
            if n_iter == 0:
                for kk in range(len(nij)):
                    if nij[kk] != j:
                        X *= np.tanh(0.5 * Lc[nij[kk]])
            else:
                for kk in range(len(nij)):
                    if nij[kk] != j:
                        X *= np.tanh(0.5 * Lq[i, nij[kk]])
            num = 1 + X
            denom = 1 - X
            for ll in range(n_messages):
                if num[ll] == 0:
                    Lr[i, j, ll] = -1
                elif denom[ll] == 0:
                    Lr[i, j, ll] = 1
                else:
                    Lr[i, j, ll] = np.log(num[ll] / denom[ll])

    # step 2 : Vertical
    for j in range(n):
        # mj = nodes[j]
        ff = nodes_hist[j]
        mj = nodes_values[nodes_counter: nodes_counter + ff]
        nodes_counter += ff
        for i in mj:
            mji = mj[:]
            Lq[i, j] = Lc[j]

            for kk in range(len(mji)):
                if mji[kk] != i:
                    Lq[i, j] += Lr[mji[kk], j]

    # LLR a posteriori:
    L_posteriori = np.zeros((n, n_messages))
    nodes_counter = 0
    for j in range(n):
        ff = nodes_hist[j]
        mj = nodes_values[nodes_counter: nodes_counter + ff]
        nodes_counter += ff
        L_posteriori[j] = Lc[j] + Lr[mj, j].sum(axis=0)

    return Lq, Lr, L_posteriori


def decode_post(H, y, snr, maxiter=1000):
    """Decode a Gaussian noise corrupted n bits message using BP algorithm.

    Decoding is performed in parallel if multiple codewords are passed in y.

    Parameters
    ----------
    H: array (n_equations, n_code). Decoding matrix H.
    y: array (n_code, n_messages) or (n_code,). Received message(s) in the
        codeword space.
    maxiter: int. Maximum number of iterations of the BP algorithm.

    Returns
    -------
    tuple (success,x)
          where
            x: posteriori array
    """
    m, n = H.shape

    bits_hist, bits_values, nodes_hist, nodes_values = _bitsandnodes(H)

    _n_bits = np.unique(H.sum(0))
    _n_nodes = np.unique(H.sum(1))

    solver = _logbp_numba

    var = 10 ** (-snr / 10)

    if y.ndim == 1:
        y = y[:, None]
    # step 0: initialization

    Lc = 2 * y / var
    _, n_messages = y.shape

    Lq = np.zeros(shape=(m, n, n_messages))

    Lr = np.zeros(shape=(m, n, n_messages))
    for n_iter in range(maxiter):
        Lq, Lr, L_posteriori = solver(bits_hist, bits_values, nodes_hist,
                                      nodes_values, Lc, Lq, Lr, n_iter)
        x = np.array(L_posteriori <= 0).astype(int)
        product = _incode(H, x)
        if product:
            break

    # x = x.squeeze()
    if n_iter == maxiter - 1:
        # warnings.warn("""Decoding stopped before convergence. You may want
        #                to increase maxiter""")
        return False, np.squeeze(L_posteriori)
    return True, np.squeeze(L_posteriori)

# ---------------------------------------------------------------------------

import ldpc96_cfg as cfg

LDPC_G = np.array(cfg.LDPC_G)
LDPC_H = np.array(cfg.LDPC_H)

def l96_encode(b50):
    return [ int(x % 2) for x in LDPC_G.dot(b50) ]

def l96_decode(llr96):
    # b96 = [ 4. if b else -4. for b in b96 ]
    r = decode_post(LDPC_H, np.array(llr96), 0, maxiter=200)
    return [ 1 if x > 0 else 0 for x in r[1] ]

def l96_data_from_code(cw):
    rtG, rx = _gausselimination(cfg.LDPC_G, cw)
    rtG = np.array(rtG)

    n, k = LDPC_G.shape
    message = [0] * k
    message[k - 1] = rx[k - 1]
    for i in reversed(range(k - 1)):
        message[i] = rx[i]
        message[i] -= rtG[i, list(range(i+1, k))].dot(
            [ message[x] for x in range(i+1, k) ])

    return [ int(x) % 2 for x in message ]

# eof
