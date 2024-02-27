#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
import logging
from pathlib import Path

import regex as re

from humaze import read_counts
from humaze.distractors import Distractors


# Google n-grams: 23688414489 -> 34.46
# Emlam:             36619096 -> 25.13
# WC2:              796907873 -> 29.57
# and/és: 33.36 / 23.11 / 27.34
# Google WC above 2^13:       340194 (threshold: 8192)
# Equivalent Emlam threshold:  35
# Equivalent WC2 threshold:   657
# Google WC above 2^25:       1203
# Equivalent Emlam threshold:  30207 / 2^15 (-9 => 2^16)
# Equivalent WC2 threshold:   672334  / 2^19 (-9 => 2^20)


def parse_arguments():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('counts_file', type=Path,
                        help='the input counts file. A headerless tsv file '
                             'with two columns: the word and its frequency.')
    parser.add_argument('output_file', type=Path,
                        help='the output .json file.')
    parser.add_argument('--count-threshold', '-c', type=int, default=1,
                        help='words below this frequency are dropped.')
    parser.add_argument('--max-bin', '-m', type=int, default=25,
                        help='words above a frequency of 2 to the power of '
                             'this number are put in the same bin. The '
                             'default is 25, which comes from the A-maze '
                             'paper and is only suitable for the Google Book '
                             'Ngram Corpus.')
    parser.add_argument('--min-word-length', type=int, default=3,
                        help='the minimum word length. Words below this '
                             'threshold are considered as if they were this '
                             'long. The default is 3.')
    parser.add_argument('--max-word-length', type=int, default=15,
                        help='the maximum word length. Words above this '
                             'threshold are considered as if they were this '
                             'long. The default is 15.')
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

    wordp = re.compile(r'([a-záéíóöőúüű]+?)(?:-e)?')
    distractors = Distractors(max_bin=args.max_bin,
                              min_word_length=args.min_word_length,
                              max_word_length=args.max_word_length)
    for word, count in read_counts(args.counts_file):
        if not wordp.fullmatch(word):
            continue
        if not count >= args.count_threshold:
            continue
        if not (len(word) > 1 or word in {'a', 's', 'ő'}):
            continue
        distractors.add(word, count)

    distractors.save_to_file(args.output_file)


if __name__ == '__main__':
    main()
