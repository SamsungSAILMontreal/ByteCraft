# Modified from https://github.com/OpenGG/swfzip/blob/master/swfzip.py
# This script is inspired by jspiro's swf2lzma repo: https://github.com/jspiro/swf2lzma
#
#
# SWF Formats:
## ZWS(LZMA)
## | 4 bytes       | 4 bytes    | 4 bytes       | 5 bytes    | n bytes    | 6 bytes         |
## | 'ZWS'+version | scriptLen  | compressedLen | LZMA props | LZMA data  | LZMA end marker |
##
## scriptLen is the uncompressed length of the SWF data. Includes 4 bytes SWF header and
## 4 bytes for scriptLen itself
##
## compressedLen does not include header (4+4+4 bytes) or lzma props (5 bytes)
## compressedLen does include LZMA end marker (6 bytes)
#
import os
import pylzma
import sys
import struct
import zlib
import bitstring

def unzip(inData):
    failed = False

    if inData[0] == ord('C'):
        # zlib SWF
        zobj = zlib.decompressobj()  # obj for decompressing data streams that won’t fit into memory at once.
        decompressData = zobj.decompress(inData[8:])
    elif inData[0] == ord('Z'):
        # lzma SWF
        decompressData = pylzma.decompress(inData[12:])
    elif inData[0] == ord('F'):
        # uncompressed SWF
        decompressData = inData[8:]
    else:
        raise Exception('not a SWF file')

    sigSize = struct.unpack("<I", inData[4:8])[0]

    decompressSize = len(decompressData) + 8

    #assert sigSize == decompressSize # 'Length not correct, decompression failed'
    if sigSize != decompressSize:
        failed = False
        print('Length not correct (We will just replace the wrong size on the .swf)')
        assert inData[0:4] + inData[4:8] == inData[0:8]
        assert inData[0:4] + struct.pack("<I", sigSize) == inData[0:8]
        header_ = inData[0:4] + struct.pack("<I", decompressSize)
    else:
        header_ = inData[0:8]
    header = list(struct.unpack("<8B", header_))
    header[0] = ord('F')

    #Generating uncompressed data
    return struct.pack("<8B", *header)+decompressData, failed

def zip(inData, compression):
    if(compression == 'lzma'):
        if inData[0] == ord('Z'): # already lzma compressed
            return None, False

        rawSwf, failed = unzip(inData);

        # 'Compressing with lzma'
        compressData = pylzma.compress(rawSwf[8:], eos=1)
        # 5 accounts for lzma props

        compressSize = len(compressData) - 5

        header = list(struct.unpack("<12B", inData[0:12]))
        header[0]  = ord('Z')
        header[3]  = header[3]>=13 and header[3] or 13
        header[8]  = (compressSize)       & 0xFF
        header[9]  = (compressSize >> 8)  & 0xFF
        header[10] = (compressSize >> 16) & 0xFF
        header[11] = (compressSize >> 24) & 0xFF

        # 'Packing lzma header'
        headerBytes = struct.pack("<12B", *header);
    else:
        if inData[0] == ord('C'): # already zlib compressed
            return None, False

        rawSwf, failed = unzip(inData);

        # 'Compressing with zlib'
        compressData = zlib.compress(rawSwf[8:])

        compressSize = len(compressData)

        header = list(struct.unpack("<8B", inData[0:8]))
        header[0] = ord('C')
        header[3]  = header[3]>=6 and header[3] or 6

        # 'Packing zlib header'
        headerBytes = struct.pack("<8B", *header)

    # 'Generating compressed data'
    return headerBytes+compressData, failed

def swf_zip_unzip(infile, outfile, operation='unzip', compression='zlib'):
    fi = open(infile, "rb")
    infileSize = os.path.getsize(infile)
    inData = fi.read()
    fi.close()
    assert inData[1] == ord('W') and inData[2] == ord('S') # "not a SWF file"

    if operation == 'unzip':
        outData, failed = unzip(inData)
    else:
        compression = compression == 'lzma' and 'lzma' or 'zlib'
        outData, failed = zip(inData, compression)

    if failed:
        return None
    if outData is None:
        outData = inData
    fo = open(outfile, 'wb')
    fo.write(outData)
    fo.close()

def get_encryption_type(infile):
    fi = open(infile, "rb")
    infileSize = os.path.getsize(infile)
    inData = fi.read()
    fi.close()
    assert inData[1] == ord('W') and inData[2] == ord('S') #"not a SWF file"

    if inData[0] == ord('C'):
        return "zlib"
    elif inData[0] == ord('Z'):
        return "lzma"
    elif inData[0] == ord('F'):
        return "uncompressed"

# taken from https://gist.github.com/nathan-osman/d86b9877221367e63088
def get_header_info_from_swf(swf_loc):
    with open(swf_loc, 'rb') as f:
        stream = bitstring.BitStream(bytes=f.read())
        s1 = chr(stream.read('uint:8')) # ord('F'), ord('C'), ord('Z') uncompressed, zlib, LZMA
        s2 = chr(stream.read('uint:8'))
        s3 = chr(stream.read('uint:8'))
        version = stream.read('uint:8')
        file_length = stream.read('uintle:32')
        if s1 == 'C': # zlib
            compression = 'ZLIB'
            stream = bitstring.BitStream(
                            bytes=zlib.decompress(stream.read('bytes'))
                        )
        elif s1 == 'Z': # lzma
            compression = 'LZMA'
            stream.bytepos += 4
            stream = bitstring.BitStream(
                            bytes=pylzma.decompress(stream.read('bytes'))
                        )
        else:
            compression = 'Uncompressed'
        nbits = stream.read('uint:5')
        xmin = stream.read('int:%d' % nbits)
        xmax = stream.read('int:%d' % nbits)
        ymin = stream.read('int:%d' % nbits)
        ymax = stream.read('int:%d' % nbits)
        w = abs(xmax - xmin) / 20
        h = abs(ymax - ymin) / 20
        stream.bytealign()
        b = stream.readlist('uintle:%d,uintle:%d' % (8, 8))
        frame_rate = float('%d.%d' % tuple(reversed(b)))
        frame_count = stream.read('uintle:16')
    return compression, file_length, version, w, h, frame_rate, frame_count

# Pseudo-check of validity, not guaranteed to be valid, but its better than nothing
def check_if_valid(infile):
    fi = open(infile, "rb")
    infileSize = os.path.getsize(infile)
    inData = fi.read()
    fi.close()
    if inData[1] == ord('W') and inData[2] == ord('S'):
        try:
            get_header_info_from_swf(infile)
            return True
        except:
            return False
    else:
        return False
