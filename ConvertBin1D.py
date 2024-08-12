import sys
import logging
import argparse
import struct
import os
import fnmatch
import multiprocessing
from multiprocessing import Process
from scipy.io import mmread, mminfo

# Assumed sizes of various C data types (in bytes)
SIZE_INT = 4
SIZE_DOUBLE = 8
SIZE_LONG_LONG = 8


def writeBUBinPartSingle(
        partfile,
        nr,
        nc,
        nnz,
        matrows,
        matvals,
        nparts
):
    logging.info('Reading part file %s' % partfile)
    f = open(partfile)
    part = [int(line) for line in f]
    f.close()
    logging.info('Completed reading partfile')

    logging.info('Computing row and nnz counts...')
    rc = [0] * (nparts)
    nnzc = [0] * (nparts)
    ia = [[0] for p in range(nparts)]
    rowids = [None] * nparts
    for r in range(nr):
        rc[part[r]] += 1
        nnzc[part[r]] += len(matrows[r])
        ia[part[r]].append(len(matrows[r]))

    ia = [[] for p in range(nparts)]
    for p in range(nparts):
        ia[p] = [0] * (rc[p] + 1)
        rowids[p] = [None] * rc[p]

    iac = [1] * nparts
    for r in range(nr):
        p = part[r]
        ia[p][iac[p]] = ia[p][iac[p] - 1] + len(matrows[r])
        rowids[p][iac[p] - 1] = r
        iac[p] += 1

    logging.info('Writing header and start location information...')
    fb = open(partfile + '.bin', 'wb')
    fb.write(struct.pack('i', nr))
    fb.write(struct.pack('i', nc))
    slocs = [None] * (nparts)
    slocs[0] = SIZE_INT + SIZE_INT + (SIZE_LONG_LONG * nparts)
    fb.write(struct.pack('q', slocs[0]))
    for p in range(1, nparts):
        psize = (2 * SIZE_INT) + ((rc[p - 1] + 1) * SIZE_INT) + \
                (nnzc[p - 1] * SIZE_INT) + (nnzc[p - 1] * SIZE_DOUBLE)
        slocs[p] = slocs[p - 1] + psize
        fb.write(struct.pack('q', slocs[p]))

    logging.info('Writing local processor matrix information...')
    for p in range(nparts):
        assert rc[p] + 1 == len(ia[p])
        fb.write(struct.pack('i', rc[p]))
        fb.write(struct.pack('i', nnzc[p]))
        fb.write(struct.pack('i' * len(ia[p]), *(ia[p])))
        for r in rowids[p]:
            fb.write(struct.pack('i' * len(matrows[r]), *(matrows[r])))
        for r in rowids[p]:
            # vals = [1.00] * len(matrows[r])
            vals = matvals[r]
            fb.write(struct.pack('d' * len(vals), *vals))

    logging.info('Completed writing')

    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mmfile', help='matrix market file')
    parser.add_argument('scheme', help='the respective directory should '
                                       'exist under matrix dir')

    args = parser.parse_args()

    # multiprocessing.log_to_stderr(logging.DEBUG)
    logging.basicConfig(format='%(processName)s %(asctime)s %(message)s',
                        level=logging.DEBUG,
                        stream=sys.stdout)

    nr, nc, nnz, _, _, _ = mminfo(args.mmfile)
    print(nr, nc, nnz)
    mat = mmread(args.mmfile)
    matrows = [[] for r in range(nr)]
    matvals = [[] for r in range(nr)]
    for (r, c, v) in zip(mat.row, mat.col, mat.data):
        matrows[r].append(c)
        matvals[r].append(v)
    # get mmfile's name
    base_name = args.mmfile.split('/')[-1]
    base_name = base_name.split('.')[0]
    # enabled multiprocessing
    path = os.path.abspath(args.mmfile)
    path = os.path.dirname(path)
    path = os.path.join(path, args.scheme)

    match_regex = f"{base_name}.inpart*[!.bin]"
    print(f"Looking for files with regex: {match_regex}")
    pfiles = [os.path.join(dirpath, f)
              for dirpath, dirnames, files in os.walk(path)
              for f in fnmatch.filter(files, match_regex)]
    print(f"Found {len(pfiles)} files")

    for i in range(len(pfiles)):
        # either core count is after inpart. or after reduced.
        try:
            K = int(pfiles[i].split('inpart.')[1])
        except:
            K = int(pfiles[i].split('reduced.')[1])
        writeBUBinPartSingle(pfiles[i], nr, nc, nnz, matrows, matvals, K)

    return


if __name__ == '__main__':
    sys.exit(main())
