#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
import logging
from pathlib import Path
import regex as re

import torch

from humaze.distractors import Distractors
from humaze.model import ProbabilityModel


def parse_arguments():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('model_name',
                        help='the language model to load.')
    parser.add_argument('distractors', type=Path,
                        help='the file that contains the distractor mappings.')
    parser.add_argument('--threshold', '-t', type=float, default=-21,
                        help='the distractor selection threshold. The '
                             'paper default is -21.')
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


pm = None
results = None


END_PUNCT = re.compile(r'^(.+?)([[:punct:]]*)$')


def reformat_distractor(word: str, distractor: str) -> str:
    """
    Formats _distractor_ according to how _word_ "looks": capitalization and
    end punctuation will match those of the latter.
    """
    word_proper, punct = END_PUNCT.fullmatch(word).groups()
    if word_proper.istitle():
        distractor = distractor.title()
    return distractor + punct


def choose_distractor(
    candidates: list[str], log_probs: torch.tensor, threshold: float
) -> str:
    """
    Picks the first candidate whose log probability is below _threshold_. If
    none is found, the word with the lowest probability is returned.
    """
    for candidate, log_prob in zip(candidates, log_probs):
        if log_prob < threshold:
            return candidate
    logging.info("Didn't found a word surprising enough; "
                 'returning least probable...')
    return candidates[log_probs.argmin()]


def main():
    args = parse_arguments()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(process)s - %(levelname)s - %(message)s'
    )

    db = Distractors.load_from_file(args.distractors)
    device = f'cuda:{args.gpu}' if args.gpu is not None else None
    global pm
    pm = ProbabilityModel.from_pretrained(
        args.model_name, args.batch_size,  device
    )

    sentence = 'Elmesélek egy történetet a világ hajnaláról:'
    prompt = {'input_ids': torch.tensor([[]], dtype=int).to(pm.model.device),
              'attention_mask': torch.tensor([[]], dtype=int).to(pm.model.device)}
    for word, tokens in pm.split_text_into_token_groups(sentence):
        candidates = db.get_candidates(word, 10)
        print(f'Distractors for {word}: {", ".join(candidates)}')
        for k, v in prompt.items():
            prompt[k] = torch.cat([v, tokens[k]], 1)
        global results
        log_probs = pm.distractor_probabilities(prompt, candidates)

        distractor = choose_distractor(candidates, log_probs, args.threshold)
        print(f'{word} -> {reformat_distractor(word, distractor)}')



if __name__ == '__main__':
    main()
