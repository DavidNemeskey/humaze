#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections.abc import Generator
import bz2
import gzip
import io
from pathlib import Path

try:
    import idzip
except ImportError:
    idzip = None


def openall(
    filename: Path | str, mode='rt', encoding=None, errors=None,
    newline=None, buffering=-1, closefd=True, opener=None,  # for open()
    compresslevel=5,  # faster default compression
):
    """
    Opens all file types known to the Python SL. There are some differences
    from the stock functions:
    - the default mode is 'rt'
    - the default compresslevel is 5, because e.g. gzip does not benefit a lot
      from higher values, only becomes slower.
    """
    filename = str(filename)
    if filename.endswith('.dz') and idzip:
        # Unfortunately idzip's API is not very good
        f = idzip.open(filename, mode.replace('t', '').replace('b', '') + 'b')
        if 't' in mode:
            return io.TextIOWrapper(f, encoding, errors,
                                    newline, write_through=True)
        else:
            return f
    elif filename.endswith('.gz') or filename.endswith('.dz'):
        # .dz is .gz, so if we don't have idzip installed, we can still read it
        return gzip.open(filename, mode, compresslevel,
                         encoding, errors, newline)
    elif filename.endswith('.bz2'):
        return bz2.open(filename, mode, compresslevel,
                        encoding, errors, newline)
    else:
        return open(filename, mode, buffering, encoding, errors, newline,
                    closefd, opener)


def read_counts(counts_file) -> Generator[tuple[str, int]]:
    """Enumerates a word count file."""
    with openall(counts_file, 'rt') as inf:
        for line in inf:
            ngram, count = line.strip().split('\t')
            yield ngram, int(count)
