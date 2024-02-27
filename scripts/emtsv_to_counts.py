#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from collections import Counter
import json
import logging
from multiprocessing import Pool
import os
from pathlib import Path

from humaze import openall, otqdm


def parse_arguments():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('input_directory', type=Path,
                        help='the input directory with the emtsv CoNLL files.')
    parser.add_argument('output_file', type=Path,
                        help='the output counts file.')
    parser.add_argument('--processes', '-P', type=int, default=1,
                        help='number of worker processes to use (max is the '
                             'num of cores, default: 1)')
    parser.add_argument('--log-level', '-L', type=str, default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help='the logging level.')

    args = parser.parse_args()
    num_procs = len(os.sched_getaffinity(0))
    if args.processes < 1 or args.processes > num_procs:
        parser.error('Number of processes must be between 1 and {}'.format(
            num_procs))
    return args


def read_emtsv_header(header: str) -> tuple[int, int]:
    """Returns the indices of the ``form`` and ``lemma`` fields."""
    d = {field: fid for fid, field in
         enumerate(header.rstrip('\n').split('\t'))}
    return d['form'], d['lemma'], d['anas']


def grep_file(input_file: Path):
    try:
        c = Counter()
        with openall(input_file, 'rt') as inf:
            form_id, lemma_id, anas_id = read_emtsv_header(inf.readline())
            for line in (l.rstrip('\n') for l in inf):
                if line and not line.startswith('# '):
                    fields = line.split('\t')
                    # Is it a common word and is it a Hungarian word at all?
                    if fields[lemma_id].islower() and json.loads(fields[anas_id]):
                        c[fields[form_id].lower()] += 1
        return c
    except:
        logging.exception(f'Error in {input_file}:')
        raise


def main():
    args = parse_arguments()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(process)s - %(levelname)s - %(message)s'
    )

    os.nice(20)

    input_files = [f for f in args.input_directory.iterdir()
                   if not f.name.startswith('.')]
    logging.info(f'Counting {len(input_files)} files...')

    with Pool(args.processes) as pool:
        counter = Counter()
        for c in otqdm(
            pool.imap_unordered(grep_file, input_files),
            f'Grepping words from {args.input_directory}...',
            total=len(input_files)
        ):
            counter.update(c)
        pool.close()
        pool.join()

    logging.info(f'Counting done. Writing output file {args.output_file}...')

    with openall(args.output_file, 'wt') as outf:
        for word, freq in sorted(counter.items(),
                                 key=lambda kv: (-kv[1], kv[0])):
            print(f'{word}\t{freq}', file=outf)

    logging.info('Done.')


if __name__ == '__main__':
    main()
