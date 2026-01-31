#!/usr/bin/env python3

# ft8_coding.py

# FT8 bit message en/decoding, extracted from:
#   https://github.com/rtmrtmrtmrtm/weakmon
#   Robert Morris, AB1HL, 2019
#   code license: MIT

# Christian Tschudin, K6CFT, 2025

import numpy as np

# FT8 bit message en/decoding (CRC and LDPC)
#   77 bits -->
#     append 14 bits CRC (for 91 bits) -->
#       LDPC(174,91) -->
#         yields 174 bits


class FT8_CODING:

    # the CRC-14 polynomial, from wsjt-x's 0x2757, with leading 1 bit
    crc14poly = [ 1,   1, 0,   0, 1, 1, 1,   0, 1, 0, 1,   0, 1, 1, 1 ]

    # LDPC tables and pack/unpack details from wsjt-x 2.0

    # this is the LDPC(174,91) parity check matrix.
    # each row describes one parity check.
    # each number is an index into the codeword (1-origin).
    # the codeword bits mentioned in each row must xor to zero.
    # From WSJT-X's ldpc_174_91_c_reordered_parity.f90
    nmx = np.array([
      [   4,  31,  59,  91,  92,  96, 153 ],
      [   5,  32,  60,  93, 115, 146,   0 ],
      [   6,  24,  61,  94, 122, 151,   0 ],
      [   7,  33,  62,  95,  96, 143,   0 ],
      [   8,  25,  63,  83,  93,  96, 148 ],
      [   6,  32,  64,  97, 126, 138,   0 ],
      [   5,  34,  65,  78,  98, 107, 154 ],
      [   9,  35,  66,  99, 139, 146,   0 ],
      [  10,  36,  67, 100, 107, 126,   0 ],
      [  11,  37,  67,  87, 101, 139, 158 ],
      [  12,  38,  68, 102, 105, 155,   0 ],
      [  13,  39,  69, 103, 149, 162,   0 ],
      [   8,  40,  70,  82, 104, 114, 145 ],
      [  14,  41,  71,  88, 102, 123, 156 ],
      [  15,  42,  59, 106, 123, 159,   0 ],
      [   1,  33,  72, 106, 107, 157,   0 ],
      [  16,  43,  73, 108, 141, 160,   0 ],
      [  17,  37,  74,  81, 109, 131, 154 ],
      [  11,  44,  75, 110, 121, 166,   0 ],
      [  45,  55,  64, 111, 130, 161, 173 ],
      [   8,  46,  71, 112, 119, 166,   0 ],
      [  18,  36,  76,  89, 113, 114, 143 ],
      [  19,  38,  77, 104, 116, 163,   0 ],
      [  20,  47,  70,  92, 138, 165,   0 ],
      [   2,  48,  74, 113, 128, 160,   0 ],
      [  21,  45,  78,  83, 117, 121, 151 ],
      [  22,  47,  58, 118, 127, 164,   0 ],
      [  16,  39,  62, 112, 134, 158,   0 ],
      [  23,  43,  79, 120, 131, 145,   0 ],
      [  19,  35,  59,  73, 110, 125, 161 ],
      [  20,  36,  63,  94, 136, 161,   0 ],
      [  14,  31,  79,  98, 132, 164,   0 ],
      [   3,  44,  80, 124, 127, 169,   0 ],
      [  19,  46,  81, 117, 135, 167,   0 ],
      [   7,  49,  58,  90, 100, 105, 168 ],
      [  12,  50,  61, 118, 119, 144,   0 ],
      [  13,  51,  64, 114, 118, 157,   0 ],
      [  24,  52,  76, 129, 148, 149,   0 ],
      [  25,  53,  69,  90, 101, 130, 156 ],
      [  20,  46,  65,  80, 120, 140, 170 ],
      [  21,  54,  77, 100, 140, 171,   0 ],
      [  35,  82, 133, 142, 171, 174,   0 ],
      [  14,  30,  83, 113, 125, 170,   0 ],
      [   4,  29,  68, 120, 134, 173,   0 ],
      [   1,   4,  52,  57,  86, 136, 152 ],
      [  26,  51,  56,  91, 122, 137, 168 ],
      [  52,  84, 110, 115, 145, 168,   0 ],
      [   7,  50,  81,  99, 132, 173,   0 ],
      [  23,  55,  67,  95, 172, 174,   0 ],
      [  26,  41,  77, 109, 141, 148,   0 ],
      [   2,  27,  41,  61,  62, 115, 133 ],
      [  27,  40,  56, 124, 125, 126,   0 ],
      [  18,  49,  55, 124, 141, 167,   0 ],
      [   6,  33,  85, 108, 116, 156,   0 ],
      [  28,  48,  70,  85, 105, 129, 158 ],
      [   9,  54,  63, 131, 147, 155,   0 ],
      [  22,  53,  68, 109, 121, 174,   0 ],
      [   3,  13,  48,  78,  95, 123,   0 ],
      [  31,  69, 133, 150, 155, 169,   0 ],
      [  12,  43,  66,  89,  97, 135, 159 ],
      [   5,  39,  75, 102, 136, 167,   0 ],
      [   2,  54,  86, 101, 135, 164,   0 ],
      [  15,  56,  87, 108, 119, 171,   0 ],
      [  10,  44,  82,  91, 111, 144, 149 ],
      [  23,  34,  71,  94, 127, 153,   0 ],
      [  11,  49,  88,  92, 142, 157,   0 ],
      [  29,  34,  87,  97, 147, 162,   0 ],
      [  30,  50,  60,  86, 137, 142, 162 ],
      [  10,  53,  66,  84, 112, 128, 165 ],
      [  22,  57,  85,  93, 140, 159,   0 ],
      [  28,  32,  72, 103, 132, 166,   0 ],
      [  28,  29,  84,  88, 117, 143, 150 ],
      [   1,  26,  45,  80, 128, 147,   0 ],
      [  17,  27,  89, 103, 116, 153,   0 ],
      [  51,  57,  98, 163, 165, 172,   0 ],
      [  21,  37,  73, 138, 152, 169,   0 ],
      [  16,  47,  76, 130, 137, 154,   0 ],
      [   3,  24,  30,  72, 104, 139,   0 ],
      [   9,  40,  90, 106, 134, 151,   0 ],
      [  15,  58,  60,  74, 111, 150, 163 ],
      [  18,  42,  79, 144, 146, 152,   0 ],
      [  25,  38,  65,  99, 122, 160,   0 ],
      [  17,  42,  75, 129, 170, 172,   0 ],
    ], dtype=np.int32)

    # Mn from WSJT-X's ldpc_174_91_c_reordered_parity.f90
    # each of the 174 rows corresponds to a codeword bit.
    # the numbers indicate which three parity
    # checks (rows in Nm) refer to the codeword bit.
    # 1-origin.
    mnx = np.array([
      [  16,  45,  73 ],
      [  25,  51,  62 ],
      [  33,  58,  78 ],
      [   1,  44,  45 ],
      [   2,   7,  61 ],
      [   3,   6,  54 ],
      [   4,  35,  48 ],
      [   5,  13,  21 ],
      [   8,  56,  79 ],
      [   9,  64,  69 ],
      [  10,  19,  66 ],
      [  11,  36,  60 ],
      [  12,  37,  58 ],
      [  14,  32,  43 ],
      [  15,  63,  80 ],
      [  17,  28,  77 ],
      [  18,  74,  83 ],
      [  22,  53,  81 ],
      [  23,  30,  34 ],
      [  24,  31,  40 ],
      [  26,  41,  76 ],
      [  27,  57,  70 ],
      [  29,  49,  65 ],
      [   3,  38,  78 ],
      [   5,  39,  82 ],
      [  46,  50,  73 ],
      [  51,  52,  74 ],
      [  55,  71,  72 ],
      [  44,  67,  72 ],
      [  43,  68,  78 ],
      [   1,  32,  59 ],
      [   2,   6,  71 ],
      [   4,  16,  54 ],
      [   7,  65,  67 ],
      [   8,  30,  42 ],
      [   9,  22,  31 ],
      [  10,  18,  76 ],
      [  11,  23,  82 ],
      [  12,  28,  61 ],
      [  13,  52,  79 ],
      [  14,  50,  51 ],
      [  15,  81,  83 ],
      [  17,  29,  60 ],
      [  19,  33,  64 ],
      [  20,  26,  73 ],
      [  21,  34,  40 ],
      [  24,  27,  77 ],
      [  25,  55,  58 ],
      [  35,  53,  66 ],
      [  36,  48,  68 ],
      [  37,  46,  75 ],
      [  38,  45,  47 ],
      [  39,  57,  69 ],
      [  41,  56,  62 ],
      [  20,  49,  53 ],
      [  46,  52,  63 ],
      [  45,  70,  75 ],
      [  27,  35,  80 ],
      [   1,  15,  30 ],
      [   2,  68,  80 ],
      [   3,  36,  51 ],
      [   4,  28,  51 ],
      [   5,  31,  56 ],
      [   6,  20,  37 ],
      [   7,  40,  82 ],
      [   8,  60,  69 ],
      [   9,  10,  49 ],
      [  11,  44,  57 ],
      [  12,  39,  59 ],
      [  13,  24,  55 ],
      [  14,  21,  65 ],
      [  16,  71,  78 ],
      [  17,  30,  76 ],
      [  18,  25,  80 ],
      [  19,  61,  83 ],
      [  22,  38,  77 ],
      [  23,  41,  50 ],
      [   7,  26,  58 ],
      [  29,  32,  81 ],
      [  33,  40,  73 ],
      [  18,  34,  48 ],
      [  13,  42,  64 ],
      [   5,  26,  43 ],
      [  47,  69,  72 ],
      [  54,  55,  70 ],
      [  45,  62,  68 ],
      [  10,  63,  67 ],
      [  14,  66,  72 ],
      [  22,  60,  74 ],
      [  35,  39,  79 ],
      [   1,  46,  64 ],
      [   1,  24,  66 ],
      [   2,   5,  70 ],
      [   3,  31,  65 ],
      [   4,  49,  58 ],
      [   1,   4,   5 ],
      [   6,  60,  67 ],
      [   7,  32,  75 ],
      [   8,  48,  82 ],
      [   9,  35,  41 ],
      [  10,  39,  62 ],
      [  11,  14,  61 ],
      [  12,  71,  74 ],
      [  13,  23,  78 ],
      [  11,  35,  55 ],
      [  15,  16,  79 ],
      [   7,   9,  16 ],
      [  17,  54,  63 ],
      [  18,  50,  57 ],
      [  19,  30,  47 ],
      [  20,  64,  80 ],
      [  21,  28,  69 ],
      [  22,  25,  43 ],
      [  13,  22,  37 ],
      [   2,  47,  51 ],
      [  23,  54,  74 ],
      [  26,  34,  72 ],
      [  27,  36,  37 ],
      [  21,  36,  63 ],
      [  29,  40,  44 ],
      [  19,  26,  57 ],
      [   3,  46,  82 ],
      [  14,  15,  58 ],
      [  33,  52,  53 ],
      [  30,  43,  52 ],
      [   6,   9,  52 ],
      [  27,  33,  65 ],
      [  25,  69,  73 ],
      [  38,  55,  83 ],
      [  20,  39,  77 ],
      [  18,  29,  56 ],
      [  32,  48,  71 ],
      [  42,  51,  59 ],
      [  28,  44,  79 ],
      [  34,  60,  62 ],
      [  31,  45,  61 ],
      [  46,  68,  77 ],
      [   6,  24,  76 ],
      [   8,  10,  78 ],
      [  40,  41,  70 ],
      [  17,  50,  53 ],
      [  42,  66,  68 ],
      [   4,  22,  72 ],
      [  36,  64,  81 ],
      [  13,  29,  47 ],
      [   2,   8,  81 ],
      [  56,  67,  73 ],
      [   5,  38,  50 ],
      [  12,  38,  64 ],
      [  59,  72,  80 ],
      [   3,  26,  79 ],
      [  45,  76,  81 ],
      [   1,  65,  74 ],
      [   7,  18,  77 ],
      [  11,  56,  59 ],
      [  14,  39,  54 ],
      [  16,  37,  66 ],
      [  10,  28,  55 ],
      [  15,  60,  70 ],
      [  17,  25,  82 ],
      [  20,  30,  31 ],
      [  12,  67,  68 ],
      [  23,  75,  80 ],
      [  27,  32,  62 ],
      [  24,  69,  75 ],
      [  19,  21,  71 ],
      [  34,  53,  61 ],
      [  35,  46,  47 ],
      [  33,  59,  76 ],
      [  40,  43,  83 ],
      [  41,  42,  63 ],
      [  49,  75,  83 ],
      [  20,  44,  48 ],
      [  42,  49,  57 ],
    ], dtype=np.int32)

    # LDPC generator matrix from WSJT-X's ldpc_174_91_c_generator.f90.
    # 83 rows, since LDPC(174,91) needs 83 parity bits.
    # each row has 23 hex digits, to be turned into 91 bits,
    # to be xor'd with the 91 data bits.
    rawg = [
      "8329ce11bf31eaf509f27fc", 
      "761c264e25c259335493132", 
      "dc265902fb277c6410a1bdc", 
      "1b3f417858cd2dd33ec7f62", 
      "09fda4fee04195fd034783a", 
      "077cccc11b8873ed5c3d48a", 
      "29b62afe3ca036f4fe1a9da", 
      "6054faf5f35d96d3b0c8c3e", 
      "e20798e4310eed27884ae90", 
      "775c9c08e80e26ddae56318", 
      "b0b811028c2bf997213487c", 
      "18a0c9231fc60adf5c5ea32", 
      "76471e8302a0721e01b12b8", 
      "ffbccb80ca8341fafb47b2e", 
      "66a72a158f9325a2bf67170", 
      "c4243689fe85b1c51363a18", 
      "0dff739414d1a1b34b1c270", 
      "15b48830636c8b99894972e", 
      "29a89c0d3de81d665489b0e", 
      "4f126f37fa51cbe61bd6b94", 
      "99c47239d0d97d3c84e0940", 
      "1919b75119765621bb4f1e8", 
      "09db12d731faee0b86df6b8", 
      "488fc33df43fbdeea4eafb4", 
      "827423ee40b675f756eb5fe", 
      "abe197c484cb74757144a9a", 
      "2b500e4bc0ec5a6d2bdbdd0", 
      "c474aa53d70218761669360", 
      "8eba1a13db3390bd6718cec", 
      "753844673a27782cc42012e", 
      "06ff83a145c37035a5c1268", 
      "3b37417858cc2dd33ec3f62", 
      "9a4a5a28ee17ca9c324842c", 
      "bc29f465309c977e89610a4", 
      "2663ae6ddf8b5ce2bb29488", 
      "46f231efe457034c1814418", 
      "3fb2ce85abe9b0c72e06fbe", 
      "de87481f282c153971a0a2e", 
      "fcd7ccf23c69fa99bba1412", 
      "f0261447e9490ca8e474cec", 
      "4410115818196f95cdd7012", 
      "088fc31df4bfbde2a4eafb4", 
      "b8fef1b6307729fb0a078c0", 
      "5afea7acccb77bbc9d99a90", 
      "49a7016ac653f65ecdc9076", 
      "1944d085be4e7da8d6cc7d0", 
      "251f62adc4032f0ee714002", 
      "56471f8702a0721e00b12b8", 
      "2b8e4923f2dd51e2d537fa0", 
      "6b550a40a66f4755de95c26", 
      "a18ad28d4e27fe92a4f6c84", 
      "10c2e586388cb82a3d80758", 
      "ef34a41817ee02133db2eb0", 
      "7e9c0c54325a9c15836e000", 
      "3693e572d1fde4cdf079e86", 
      "bfb2cec5abe1b0c72e07fbe", 
      "7ee18230c583cccc57d4b08", 
      "a066cb2fedafc9f52664126", 
      "bb23725abc47cc5f4cc4cd2", 
      "ded9dba3bee40c59b5609b4", 
      "d9a7016ac653e6decdc9036", 
      "9ad46aed5f707f280ab5fc4", 
      "e5921c77822587316d7d3c2", 
      "4f14da8242a8b86dca73352", 
      "8b8b507ad467d4441df770e", 
      "22831c9cf1169467ad04b68", 
      "213b838fe2ae54c38ee7180", 
      "5d926b6dd71f085181a4e12", 
      "66ab79d4b29ee6e69509e56", 
      "958148682d748a38dd68baa", 
      "b8ce020cf069c32a723ab14", 
      "f4331d6d461607e95752746", 
      "6da23ba424b9596133cf9c8", 
      "a636bcbc7b30c5fbeae67fe", 
      "5cb0d86a07df654a9089a20", 
      "f11f106848780fc9ecdd80a", 
      "1fbb5364fb8d2c9d730d5ba", 
      "fcb86bc70a50c9d02a5d034", 
      "a534433029eac15f322e34c", 
      "c989d9c7c3d3b8c55d75130", 
      "7bb38b2f0186d46643ae962", 
      "2644ebadeb44b9467d1f42c", 
      "608cc857594bfbb55d69600"
    ]

    # gen[row][col], derived from rawg, has one row per
    # parity bit, to be xor'd with the 91 data bits.
    # thus gen[83][91].
    # as in encode174_91.f90
    gen = []

    def __init__(self):
        # turn rawg into gen.
        assert len(self.rawg) == 83

        if len(self.gen) == 0:
            hex2 = { hex(i)[2]:i for i in range(16) }
            for e in self.rawg:
                row = np.zeros(91, dtype=np.int32)
                for i,c in enumerate(e):
                    x = hex2[c]
                    for j in range(0, 4):
                        ind = i*4 + (3-j)
                        if ind >= 0 and ind < 91:
                            if (x & (1 << j)) != 0:
                                row[ind] = 1
                            else:
                                row[ind] = 0
                self.gen.append(row)

        # turn gen[] into a systematic array by prepending
        # a 91x91 identity matrix.
        self.gen_sys = np.zeros((174, 91), dtype=np.int32)
        self.gen_sys[91:,:] = self.gen
        self.gen_sys[0:91,:] = np.eye(91, dtype=np.int32)


    def crc14(self, a77):
        # https://gist.github.com/evansneath/4650991
        div = self.crc14poly
        divlen = len(div)
        code = [0] * (divlen-1)
        msg = np.append(a77, code)

        # loop over every message bit (minus the appended code)
        for i in range(len(msg)-len(code)):
            # If that messsage bit is 1, perform modulo 2 multiplication
            if msg[i] == 1:
                msg[i:i+divlen] = np.mod(msg[i:i+divlen] + div, 2)

        # output the error-checking code portion of the message generated
        return [x for x in msg[-len(code):]]

    def check_crc14(self, a91):
        cksum = self.crc14(a91[0:77])
        return np.array_equal(cksum, a91[-14:])

    def ldpc_check(self, codeword):
        # does a 174-bit codeword pass the LDPC parity checks?
        assert len(codeword) == 174

        for e in self.nmx:
            x = 0
            for i in e:
                if i != 0:
                    x ^= codeword[i-1]
            if x != 0:
                return False
        return True

    def ldpc_parity(self, codeword):
        # does a 174-bit codeword pass the LDPC parity checks?
        assert len(codeword) == 174

        cnt = 0
        for e in self.nmx:
            x = 0
            for i in e:
                if i != 0:
                    x ^= codeword[i-1]
            if x == 0:
                cnt += 1
        return cnt

    def ldpc_encode(self, a91):
        # a91 is 91 bits of plain-text; returns a 174-bit codeword (0/1)
        # mimics wsjt-x's encode174_91.f90.
        assert len(a91) == 91

        a91 = np.array(a91, dtype=np.int32)
        cw = np.zeros(174, dtype=np.int32)
        np.dot(self.gen_sys[91:,:], a91, out=cw[91:])
        np.mod(cw[91:], 2, out=cw[91:])
        cw[0:91] = a91

        return cw

    def ldpc_decode(self, llr174, max_iters):
        # given a 174-bit codeword as an array of log-likelihood ratios,
        # return [ nok, plain ], where nok is the number of parity
        #        checks that worked out, should be 83=174-91.

        # LLR encoding: codeword[i] = log ( P(x=0) / P(x=1) )
        # typical: -4.5 is 'sure 1', 4.5 is 'sure 0'

        # this is an implementation of the sum-product algorithm
        # from Sarah Johnson's Iterative Error Correction book.

        # 174 codeword bits:
        #   91 systematic data bits
        #   83 parity checks

        # Mji
        # each llr174 bit i tells each parity check j
        # what the bit's log-likelihood of being 0 is
        # based on information *other* than from that
        # parity check.
        m = np.zeros((83, 174))

        for i in range(174):
            for j in range(83):
                m[j][i] = llr174[i]

        for iter in range(max_iters):
            # Eji
            # each check j tells each codeword bit i the
            # log likelihood of the bit being zero based
            # on the *other* bits in that check.
            e = np.zeros((83, 174))

            # messages from checks to bits.
            # for each parity check
            #for j in range(0, 83):
            #    # for each bit mentioned in this parity check
            #    for i in Nm[j]:
            #        if i <= 0:
            #            continue
            #        a = 1
            #        # for each other bit mentioned in this parity check
            #        for ii in Nm[j]:
            #            if ii != i:
            #                a *= math.tanh(m[j][ii-1] / 2.0)
            #        e[j][i-1] = math.log((1 + a) / (1 - a))
            for i in range(7):
                a = np.ones(83)
                for ii in range(7):
                    if ii != i:
                        x1 = np.tanh(m[range(0, 83), self.nmx[:,ii]-1] / 2.0)
                        x2 = np.where(np.greater(self.nmx[:,ii], 0.0), x1, 1.0)
                        a = a * x2
                # avoid divide by zero, i.e. a[i]==1.0
                # XXX why is a[i] sometimes 1.0?
                b = np.where(np.less(a, 0.99999), a, 0.99)
                c = np.log((b + 1.0) / (1.0 - b))
                # have assign be no-op when nmx[a,b] == 0
                d = np.where(np.equal(self.nmx[:,i], 0),
                                e[range(83), self.nmx[:,i]-1],
                                c)
                e[range(83), self.nmx[:,i]-1] = d

            # decide if we are done -- compute the corrected codeword,
            # see if the parity check succeeds.
            # sum the three log likelihoods contributing to each codeword bit.
            e0 = e[self.mnx[:,0]-1, range(174)]
            e1 = e[self.mnx[:,1]-1, range(174)]
            e2 = e[self.mnx[:,2]-1, range(174)]
            ll = llr174 + e0 + e1 + e2
            # log likelihood > 0 => bit=0.
            cw = [ 1 if x < 0 else 0 for x in ll ]
            if self.ldpc_check(cw): # success!
                return (91, cw)

            # messages from bits to checks.
            for j in range(0, 3):
                # for each column in Mn.
                ll = llr174
                if j != 0:
                    e0 = e[self.mnx[:,0]-1, range(0,174)]
                    ll = ll + e0
                if j != 1:
                    e1 = e[self.mnx[:,1]-1, range(0,174)]
                    ll = ll + e1
                if j != 2:
                    e2 = e[self.mnx[:,2]-1, range(0,174)]
                    ll = ll + e2
                m[self.mnx[:,j]-1, range(0,174)] = ll

        # could not decode.
        return [ self.ldpc_parity(cw), cw ]

    def ldpc_extract(self, codeword):
        return codeword[:91]

# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import math
    import random
    import sys
    import time

    FT8 = FT8_CODING()

    print("testing CRC14 .. ", end='')
    msg = [0] * 82
    msg[3] = 1
    msg[7] = 1
    msg[44] = 1
    msg[45] = 1
    msg[46] = 1
    msg[51] = 1
    msg[61] = 1
    msg[71] = 1

    expected = [ 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1 ]

    cksum = FT8.crc14(msg)
    eq = np.equal(cksum, expected)
    assert np.all(eq)
    print("ok")

    print("testing FT8's LDPC:")
    for max_iter in [1*17, 2*17, 4*17, 8*17]:
        tt = 0.0
        niters = 500 # 5000
        ok = 0
        for iter in range(niters):
            if iter % 20 == 0:
                print('.', end='')
            sys.stdout.flush()
            # ldpc_encode() takes 91 bits.
            a77 = np.random.randint(0, 2, 77, dtype=np.int32)
            a91 = np.append(a77, FT8.crc14(a77))
            a174 = FT8.ldpc_encode(a91)

            # check that ldpc_encode() generated the right parity bits.
            assert FT8.ldpc_check(a174)

            # turn hard bits into 0.99 vs 0.01 log-likelihood,
            # log( P(0) / P(1) )
            # log base e.
            two = np.array([ 4.6, -4.6 ])
            ll174 = two[a174]

            # check decode is perfect before wrecking bits.
            [ nn, d91 ] = FT8.ldpc_decode(ll174, max_iter)
            assert np.array_equal(a91, d91)
            assert FT8.check_crc14(d91)

            # wreck some bits
            #for junk in range(0, 70):
            #    ll174[random.randint(0, len(ll174)-1)] = (random.random() - 0.5) * 4

            perm = np.random.permutation(len(ll174))
            perm = perm[0:70]
            for i in perm:
                p = random.random()
                bit = a174[i]
                if random.random() > p:
                    # flip the bit
                    # print('@', end='')
                    bit = 1 - bit
                if bit == 0:
                    p = 0.5 + (p / 2)
                else:
                    p = 0.5 - (p / 2)
                ll = math.log(p / (1.0 - p))
                ll174[i] = ll

            t0 = time.time()

            # decode LDPC(174,91)
            [ _, d91 ] = FT8.ldpc_decode(ll174, max_iter)

            t1 = time.time()
            tt += t1 - t0

            if np.array_equal(a91, d91):
                ok += 1

        print()
        print("max_iters %d, success %.2f, %.6f sec/call" % (max_iter,
                                                             ok / float(niters),
                                                             tt / niters))

# eof
