#!/usr/bin/env python3

# golay24.py

# extended binary Golay, adapted from: https://github.com/pkdoshinji/Golay
# Jan 2026 <christian.tschudin@unibas.ch>, HB9HUH/K6CFT

'''
Golay.py: Golay encoding and decoding for transmission of data across a noisy channel
Author: Patrick Kelly
Email: patrickyunen@gmail.com
Last revised: April 2, 2020
'''

wordlength = 12

I = [[1,0,0,0,0,0,0,0,0,0,0,0],
     [0,1,0,0,0,0,0,0,0,0,0,0],
     [0,0,1,0,0,0,0,0,0,0,0,0],
     [0,0,0,1,0,0,0,0,0,0,0,0],
     [0,0,0,0,1,0,0,0,0,0,0,0],
     [0,0,0,0,0,1,0,0,0,0,0,0],
     [0,0,0,0,0,0,1,0,0,0,0,0],
     [0,0,0,0,0,0,0,1,0,0,0,0],
     [0,0,0,0,0,0,0,0,1,0,0,0],
     [0,0,0,0,0,0,0,0,0,1,0,0],
     [0,0,0,0,0,0,0,0,0,0,1,0],
     [0,0,0,0,0,0,0,0,0,0,0,1]]

B = [[1,1,0,1,1,1,0,0,0,1,0,1],
     [1,0,1,1,1,0,0,0,1,0,1,1],
     [0,1,1,1,0,0,0,1,0,1,1,1],
     [1,1,1,0,0,0,1,0,1,1,0,1],
     [1,1,0,0,0,1,0,1,1,0,1,1],
     [1,0,0,0,1,0,1,1,0,1,1,1],
     [0,0,0,1,0,1,1,0,1,1,1,1],
     [0,0,1,0,1,1,0,1,1,1,0,1],
     [0,1,0,1,1,0,1,1,1,0,0,1],
     [1,0,1,1,0,1,1,1,0,0,0,1],
     [0,1,1,0,1,1,1,0,0,0,1,1],
     [1,1,1,1,1,1,1,1,1,1,1,0]]


# Multiply two matrices over the Galois field GF2
def GF2_matrix(A,B):
    C_rows = len(A)
    C_cols = len(B[0])
    C = [[0 for k in range(C_cols)] for i in range(C_rows)]

    for i in range(C_rows):
        for k in range(C_cols):
            for j in range(len(A[0])):
                C[i][k] ^= (A[i][j] & B[j][k])
    return C


# Given the (n x i) matrix A and the (n X j) matrix B,
# return the (n x (i + j)) matrix (A|B)
def conjoin(A,B):
    columns = len(A[0]) + len(B[0])
    rows = len(A)
    conjoined = [0] * rows
    for row in range(rows):
        conjoined[row] = A[row] + B[row]
    return conjoined


# Return the transpose of a matrix
def transpose(matrix):
    t_rows = len(matrix[0])
    t_cols = len(matrix)
    transposed = [[0 for k in range(t_cols)] for i in range(t_rows)]

    for i in range(t_rows):
        for j in range(t_cols):
            transposed[i][j] = matrix[j][i]
    return transposed


# Return column n of a matrix
def get_column(matrix, col_num):
    B_col = [0] * len(matrix[0])
    for k in range(len(matrix)):
        B_col[k] = matrix[col_num][k]
    return B_col


# Add two vectors over the Galois field GF2
def add_vectors(A,B):
    length = len(A)
    C = [0]*length
    for index in range(length):
        C[index] = A[index] ^ B[index]
    return C

# Get generator matrix for the extended binary Golay code (G24)
G = conjoin(I,B)

# Get transpose of parity-check matrix for G24
Ht = transpose(conjoin(B,I))


# ---------------------------------------------------------------------------
# API

def golay_encode(b12):
    return GF2_matrix([b12],G)[0]

def golay_decode(b24):
    syndrome1 = GF2_matrix([b24], Ht)[0]
    weight = sum(syndrome1) #Weight of the syndrome

    # Case 1: if Hamming weight of syndrome <= 3, we know the error vector
    if weight <= 3:
        error_L = [0] * 12
        error = error_L + syndrome1

    # Case 2: Otherwise (wt>3), we process it further...
    else:
        S1_dict = {}
        small_weights = {}
        for j in range(12):
            sum_vec = add_vectors(syndrome1,get_column(B,j)) #S1 + Bi
            syn_weight = sum(sum_vec) #wt(S1 + Bi)
            wt = (syn_weight, sum_vec)
            if syn_weight <= 2:
                small_weights[j] = wt

        # Case 2.1: one subvector weight <= 2
        if len(small_weights) == 1:
            error = I[list(small_weights.keys())[0]] + sum_vec

        # Case 2.2: several subvector weights <= 2
        elif len(small_weights) > 1:
            weightlist = [small_weights[key][0] for key in small_weights]
            smallest = min(weightlist)
            for key in small_weights:
                if small_weights[key][0] == smallest:
                    error = small_weights[key][1][0]
                    break

        # Case 2.3: all subvector weights > 3
        else:
            # Calculate syndrome2
            syndrome2 = GF2_matrix([syndrome1], transpose(B))[0]
            weight2 = sum(syndrome2)

            # Case 2.3.1: If the weight of syndrome2 <= 3, we know the error
            if weight2 <= 3:
                error_R = [0] * 12
                error = syndrome2 + error_R

            # Case 2.3.2: Otherwise, we process further...
            else:
                small_weights = {}
                for j in range(12):
                    sum_vec = add_vectors(syndrome2, get_column(B, j))
                    syn_weight = sum(sum_vec)
                    wt = (syn_weight, sum_vec)
                    if syn_weight <= 2:
                        error = sum_vec + I[j]
                        break
                    else:
                        error = [0] * 24

    # corrected Golay word = (received vector) - (error vector)
    corrected = add_vectors(b24, error)
    # return data part = first half of corrected Golay word
    return corrected[:12]

# ---------------------------------------------------------------------------

if __name__ == '__main__':

    import os

    def vect2str(v):
        return ''.join([str(x) for x in v])

    print("----sent----     --------codeword--------     ----modified_codeword---     ----rcvd---- -ok-")
    print()

    for val in range(4096):
        b12 = [(val >> i) & 1 for i in range(12)][::-1]
        cw = golay_encode(b12)
        w2 = [x for x in cw]
        for i in range(3): # flip up to three bits
            pos = os.urandom(1)[0] % len(w2)
            w2[pos] = 1 - w2[pos]
        r = golay_decode(w2)

        b12 = vect2str(b12)
        cw  = vect2str(cw)
        w2  = vect2str(w2)
        r   = vect2str(r)
        print(b12, "-->", cw, "-->", w2, "-->", r, r == b12)

# eof
