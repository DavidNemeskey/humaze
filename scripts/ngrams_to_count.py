#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from collections import Counter
import gzip
import logging
from pathlib import Path
import re

from tqdm import tqdm


def parse_arguments():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('input_dir', type=Path,
                        help='the Google ngram directory directory')
    parser.add_argument('output_file', type=Path,
                        help='the output file in the counts format.')
    parser.add_argument('--log-level', '-L', type=str, default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help='the logging level.')
    return parser.parse_args()


def main():
    args = parse_arguments()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(process)s - %(levelname)s - %(message)s'
    )

    pos_pattern = re.compile('(?:.+)_[A-Z]+')

    ngram_counts = Counter()
    for ngram_file in tqdm(args.input_dir.iterdir(), 'Converting...'):
        if ngram_file.is_file():
            with gzip.open(ngram_file, 'rt') as inf:
                for line in inf:
                    if not pos_pattern.fullmatch(ngram) and ngram.isalpha():
                        ngram, _, count, _ = line.strip().split('\t')
                        ngram_counts[ngram] += int(count)

    if args.output_file.suffix != '.gz':
        output_file = args.output_file.with_suffix('.gz')
    else:
        output_file = args.output_file
    with gzip.open(output_file, 'wt') as outf:
        for ngram, count in ngram_counts.most_common():
            print(f'{ngram}\t{count}', file=outf)


if __name__ == '__main__':
    main()
