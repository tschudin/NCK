#!/usr/bin/env python3

# hamming84.py
# an implementation of extending Hamming(8,4) codes using lookup tables

# (C) Jan 2026 <christian.tschudin@unibas.ch> HB9HUH/K6CFT
# SW released under the MIT license

# ---------------------------------------------------------------------------
# internal

def _int_to_4bits(val):
    return [(val >> i) & 1 for i in range(4)][::-1]
    
def _int_to_8bits(val):
    return [(val >> i) & 1 for i in range(8)][::-1]

def _bits_to_int(vect):
    val = 0
    for i in range(len(vect)):
        val = (val << 1) | vect[i]
    return val

def h84_init():
    global h84_enc_lut, h84_dec_lut, h84_dec_ok

    def encode(b4):
        p1 = (b4[0]+b4[1]+b4[3]) % 2
        p2 = (b4[0]+b4[2]+b4[3]) % 2
        p4 = (b4[1]+b4[2]+b4[3]) % 2
        b7 = [p1, p2, b4[0], p4] + b4[1:]
        px = sum(b7,0) % 2 # extension parity value
        return b7 + [px]

    def decode(b8):
        b4 = h84_data_from_code(b8)
        e = encode(b4)
        p1 = 1 if e[0] != b8[0] else 0
        p2 = 1 if e[1] != b8[1] else 0
        p4 = 1 if e[3] != b8[3] else 0
        px = sum(b8[:7],0) % 2
        s = (1 if p1 else 0) + (2 if p2 else 0) + (4 if p4 else 0)
        if px != b8[7]: # one-bit error
            if s in [1, 2, 4]: # it was one of the parity bits
                return (1, b4)
            b8 = [x for x in b8]
            b8[s-1] = 1 - b8[s-1]
            return (1, h84_data_from_code(b8)) # corrected
        if (p1+p2+p4) == 0: # no errors
            return (1, b4)
        return (0, b4)  # more than one error

    h84_enc_lut = bytearray(16)
    for i in range(16):
        e = encode(_int_to_4bits(i))
        h84_enc_lut[i] = _bits_to_int(e)
    h84_enc_lut = bytes(h84_enc_lut)

    tmp_lut = bytearray(256)
    tmp_ok  = [0] * 256
    for i in range(256):
        tmp_ok[i], d = decode( _int_to_8bits(i) )
        tmp_lut[i] = _bits_to_int(d)

    h84_dec_lut = [0] * 128
    for i in range(len(h84_dec_lut)):
        h84_dec_lut[i] = (tmp_lut[2*i] << 4) | (tmp_lut[2*i+1])
    h84_dec_lut = bytes(h84_dec_lut)

    h84_dec_ok = [0] * 32
    for i in range(256):
        if tmp_ok[i]:
            h84_dec_ok[i//8] |= 1 << (i % 8)
    h84_dec_ok = bytes(h84_dec_ok)

# ---------------------------------------------------------------------------
# API

def h84_encode(b4):
    # maps a list with 4 bits to a Hamming(8,4) codeword (list of 8 bits)
    ndx = _bits_to_int(b4)
    e = h84_enc_lut[ndx]
    return _int_to_8bits(e)

def h84_decode(b8):
    # Decodes a 8 bit vector
    #   If correct, (True, <4 bit vector>) will be returned.
    #   If not correct(able), (False, <4 bit vector>) will be returned
    ndx = _bits_to_int(b8)
    ok = True if h84_dec_ok[ndx//8] & (1 << (ndx % 8)) else False
    d = h84_dec_lut[ndx//2]
    d = (d >> 4) if ndx % 2 == 0 else (d & 0x0f)
    return (ok, _int_to_4bits(d))

def h84_data_from_code(cw):
    return cw[2:3] + cw[4:7]
    
h84_init()


# ---------------------------------------------------------------------------
# demo

if __name__ == '__main__':

    # tabulate the mappings:

    print("Code book")
    codebook = {}
    for i in range(16):
        b4 = _int_to_4bits(i)
        e = h84_encode(b4)
        b = ''.join([str(x) for x in b4])
        codebook[b] = e
        print(b, "-->", ''.join([str(x) for x in e]))
    print()

    print("Decode book")
    hascodes = { bin(16+i)[3:] : [] for i in range(16) }
    for i in range(256):
        b = bin(256+i)[3:]
        b8 = [int(x) for x in b]
        ok, d = h84_decode(b8)
        b4 = ''.join([str(x) for x in d])
        print(b, "-->", b4, ok)
        if ok:
            hascodes[b4].append(''.join([str(x) for x in b8]))
    print()

    print("Valid codewords (0 or 1 bit errors)")
    for b,lst in hascodes.items():
        print(f"{b}: [ {', '.join([cw for cw in lst])} ]")

# eof
