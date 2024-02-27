#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from itertools import groupby
import logging
from operator import itemgetter
from pathlib import Path
import readline
import sys

import regex as re
import torch

from humaze import append_to_name, openall
from humaze.distractors import Distractors
from humaze.model import ProbabilityModel


def parse_arguments():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('input_file', type=Path,
                        help='the input file, one sentence per line.')
    parser.add_argument('distractors', type=Path,
                        help='the file that contains the distractor mappings.')
    parser.add_argument('--model-name', '-m',
                        help='the language model to load. If not specified, '
                             'a random word will be selected as distractor '
                             'from the list of candidates at each position.')
    parser.add_argument('--number-of-variants', '-n', type=int, default=1,
                        help='the number of output files to generate with '
                             'different distractor variations. The default is '
                             '1.')
    parser.add_argument('--addendum', '-a', default='_distractors',
                        help='the string to add to the name of the output '
                             'files. The default is "_distractors".')
    parser.add_argument('--threshold', '-t', type=float, default=-21,
                        help='the distractor selection threshold. The '
                             'paper default is -21.')
    parser.add_argument('--candidates', '-c', type=int, default=100,
                        help='the number of candidates to consider for each '
                             'word. The default is 100.')
    parser.add_argument('--batch-size', '-b', type=int, default=1,
                        help='the batch size to use when calculating '
                             'distractor probabilities. Does not make sense '
                             'to make it larger than 100. The default is 1, '
                             'which should be changed as GPU memory allows.')
    parser.add_argument('--gpu', '-g',
                        help='the id of the GPU to use. If not specified, the '
                             'code will be run on CPU.')
    parser.add_argument('--log-level', '-L', type=str, default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help='the logging level.')
    return parser.parse_args()


# We support -e, but no other punctuations, numbers, etc.
END_PUNCT = re.compile(r'^([[:alpha:]]+?)((?:-e)?[[:punct:]]*)?')


def reformat_distractor(word: str, distractor: str) -> str:
    """
    Formats _distractor_ according to how _word_ "looks": capitalization and
    end punctuation will match those of the latter.
    """
    word_proper, punct = END_PUNCT.fullmatch(word).groups()
    if word_proper.istitle():
        distractor = distractor.title()
    return distractor + punct


def normalize_word(word: str) -> str:
    """Normalizes _word_ by making it lower case and string punctuation."""
    return END_PUNCT.fullmatch(word).group(1).lower()


def choose_distractor(
    candidates: list[str], log_probs: torch.tensor, threshold: float,
    word: str, stats: list[tuple[str, int]]
) -> str:
    """
    Picks the first candidate whose log probability is below _threshold_. If
    none is found, the word with the lowest probability is returned.
    """
    arr = [(candidate, log_prob) for candidate, log_prob in
           zip(candidates, log_probs) if log_prob < threshold]
    stats.append((word, len(arr)))
    if arr:
        return arr[0][0]
    else:
        logging.info(f"Didn't found a candidate surprising enough for {word}; "
                     'returning least probable...')
        return candidates[log_probs.argmin()]
    # for candidate, log_prob in zip(candidates, log_probs):
    #     if log_prob < threshold:
    #         return candidate
    # logging.info(f"Didn't found a candidate surprising enough for {word}; "
    #              'returning least probable...')
    # return candidates[log_probs.argmin()]


def get_candidates(
    sentence: str, pm: ProbabilityModel, db: Distractors,
    threshold: float, num_candidates: int, stats: list[tuple[str, int]]
) -> str:
    distractors = []
    prompt = {'input_ids': torch.tensor([[]], dtype=int).to(pm.device()),
              'attention_mask': torch.tensor([[]], dtype=int).to(pm.device())}
    for word, tokens in pm.split_text_into_token_groups(sentence):
        logging.debug(f'ZZZ {word=} {tokens=}')
        normalized_word = normalize_word(word)
        candidates = db.get_candidates(normalized_word, num_candidates)
        logging.debug(f'Distractors for {word}: {", ".join(candidates)}')
        for k, v in prompt.items():
            logging.debug(f'{k=} / {v=}')
            prompt[k] = torch.cat([v, tokens[k]], 1)
        log_probs = pm.distractor_probabilities(prompt, candidates)

        distractor = choose_distractor(candidates, log_probs, threshold,
                                       normalized_word, stats)
        logging.info(f'DDD {distractor=}')
        distractors.append(reformat_distractor(word, distractor))
    return ' '.join(distractors)


def main():
    args = parse_arguments()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(process)s - %(levelname)s - %(message)s'
    )

    if args.model_name is None:
        reply = input('No model specified. Are you sure you want to randomly '
                      'select dispatchers from candidates? (yes/no) ')
        if reply.lower() in {'y', 'yes'}:
            logging.warning('No model is used, dispatcher will be selected '
                            'randomly!')
        else:
            sys.exit(0)

    db = Distractors.load_from_file(args.distractors)
    device = f'cuda:{args.gpu}' if args.gpu is not None else None
    pm = ProbabilityModel.from_pretrained(
        args.model_name, args.batch_size, device
    )

    if args.number_of_variants == 1:
        output_files = [append_to_name(args.input_file, args.addendum)]
    else:
        output_files = [
            append_to_name(args.input_file, f'{args.addendum}_{variant}')
            for variant in range(1, args.number_of_variants + 1)
        ]

    dict_stats = None
    for output_file in output_files:
        stats = []
        with (openall(args.input_file, 'rt') as inf,
              openall(output_file, 'wt') as outf):
            for line in map(str.strip, inf):
                if line:
                    print(get_candidates(line, pm, db, args.threshold,
                                         args.candidates, stats),
                          end='', file=outf)
                print(file=outf)
        if not dict_stats:
            stats.sort()
            avgs = []
            weighted_avgs = []
            type_stats = []
            for word, lens in groupby(stats, key=itemgetter(0)):
                lengths = [length for _, length in lens]
                sum_lengths = sum(lengths)
                weighted_avgs.append(sum_lengths)
                avgs.append(sum_lengths / len(lengths))
                type_stats.append((word, sum_lengths / len(lengths)))
            logging.info(f'Per-type average length: {sum(avgs) / len(avgs)}.')
            logging.info(f'Per-token average length: {sum(weighted_avgs) / len(stats)}.')
            type_stats.sort(key=lambda wl: (-wl[1], wl[0]))
            type_print = '\n'.join(f'{w}: {l}' for w, l in type_stats)
            logging.info(f'Type lengths:\n{type_print}')


if __name__ == '__main__':
    main()
