#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from math import floor, log2
from pathlib import Path
from random import shuffle
from typing import Optional  # , Self

from humaze import openall


class Distractors:
    """A "database" used to store distractors."""
    def __init__(
        self,
        words_to_bins: Optional[dict[str, int]] = None,
        length_bin_to_distractors: Optional[
            dict[tuple[int, int], list[str]]
        ] = None,
        max_bin: int = 25,
        min_word_length = 3,
        max_word_length = 15
    ):
        """
        Creates a new distractors database. To create an empty database, to be
        filled word-by-word with :meth:`add`, supply ``None`` for the first two
        arguments.

        :param words_to_bins: the word -> frequency bin mapping.
        :param length_bin_to_distractors: the length + bin -> words mapping.
        :param max_bin: words above a frequency of 2 to the power of this
                        number are put in the same bin. The default is 25,
                        which comes from the A-maze paper and is only suitable
                        for the Google Book Ngram Corpus.
        :param min_word_length: the minimum word length. Words below this
                                threshold are considered as if they were this
                                long.
        :param max_word_length: the maximum word length. Words above this
                                threshold are considered as if they were this
                                long.
        """
        if (words_to_bins is None) != (length_bin_to_distractors is None):
            raise ValueError(
                'Either both of words_to_bins and length_bin_to_distractors '
                'must be specified, or neither.'
            )

        self.words_to_bins = words_to_bins or {}
        self.length_bin_to_distractors = length_bin_to_distractors or {}
        self.max_bin = max_bin
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length

    def get_candidates(self, word, min_candidates: int = 100) -> list[str]:
        """
        Returns the distractor candidates for _word_. These are the words that
        have the same length and are in the same frequency bin. If there are
        fewer words in the bin than _min_candidates_, the next higher
        frequency bin is used to supplement.
        """
        candidates = []
        length = self._word_len(word)
        freq_bin = self.words_to_bins[word]
        while len(candidates) < min_candidates and freq_bin <= self.max_bin:
            new_candidates = [
                w for w in self.length_bin_to_distractors[f'{length}_{freq_bin}']
                if w != word
            ]
            shuffle(new_candidates)
            candidates.extend(new_candidates)
            freq_bin += 1
        return candidates[:min_candidates]

    def add(self, word, count):
        """Adds _word_ with frequency _count_ to the distractors database."""
        freq_bin = min(floor(log2(count)), self.max_bin)
        self.words_to_bins[word] = freq_bin
        length = self._word_len(word)
        self.length_bin_to_distractors.setdefault(
            f'{length}_{freq_bin}', []
        ).append(word)

    def _word_len(self, word) -> int:
        """
        Returns the normalized word length (considering :attr:`min_word_length`
        and :attr:`max_word_length`).
        """
        return min(
            max(self.min_word_length, len(word)), self.max_word_length
        )

    @classmethod
    def load_from_file(cls, distractors_file: Path | str) -> 'Distractors':
        """Reads the data from the JSON file _distractors_file_."""
        with openall(distractors_file, 'rt') as inf:
            data = json.load(inf)
        return Distractors(**data)

    def save_to_file(self, distractors_file: Path | str):
        """Writes the data to the JSON file _distractors_file_."""
        data = {'words_to_bins': self.words_to_bins,
                'length_bin_to_distractors': self.length_bin_to_distractors}
        with openall(distractors_file, 'wt') as outf:
            print(json.dumps(data, ensure_ascii=False, indent=4), file=outf)
