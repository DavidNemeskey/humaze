#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Code for computing distractor probabilities from transformer LLMs."""

from abc import ABC, abstractmethod
import logging
from typing import Optional

from more_itertools import chunked
import torch
from torch.distributions import Categorical
from transformers import AutoModelForCausalLM, AutoTokenizer


class ProbabilityModel(ABC):
    @abstractmethod
    def device(self):
        """Returns the device the model runs on."""
        ...

    @abstractmethod
    def distractor_probabilities(self, prompt, candidates) -> torch.Tensor:
        """
        Computes the log probabilities of all _candidates_ after _prompt_.

        :return: the log probabilities.
        """
        ...

    @abstractmethod
    def split_text_into_token_groups(
        self, text
    ) -> list[tuple[str, torch.Tensor]]:
        """
        Splits _text_ in token groups that represent words. Splitting is
        performed along whitespaces, so a preformatted text is required.

        :param text: the input text.
        :return: word, token id tensor pairs.
        """
        ...

    @staticmethod
    def from_pretrained(
        model_name: Optional[str],
        batch_size: Optional[int],
        device: Optional[str]
    ) -> 'ProbabilityModel':
        if model_name is not None:
            model = AutoModelForCausalLM.from_pretrained(model_name)  # .to('cuda')
            if device:
                model.to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                      padding_side='right')
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.pad_token = tokenizer.eos_token
            return PretrainedProbabilityModel(model, tokenizer, batch_size)
        else:
            return RandomProbabilityModel()


class PretrainedProbabilityModel(ProbabilityModel):
    def __init__(self, model, tokenizer, batch_size: int = 1):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def device(self):
        return self.model.device

    def tokenize(self, words: str | list[str]) -> dict[str, torch.Tensor]:
        """
        Tokenizes _words_ and moves the resulting tensors to the same device
        as the model.
        """
        return self.tokenizer(
            words, return_tensors='pt', padding=True
        ).to(self.model.device)

    def to_tokens(self, prompt):
        words = prompt.split(' ')
        word_tokens = [
            self.tokenizer(f' {word}' if word_id else word)
            for word_id, word in enumerate(words)
        ]
        return word_tokens

    def tokenize_to_tokens(self, text):
        """Helper function for debugging."""
        return ' '.join(
            self.tokenizer.convert_ids_to_tokens(
                self.tokenizer(text)['input_ids']
            )
        )

    def split_text_into_token_groups(
        self, text
    ) -> list[tuple[str, torch.Tensor]]:
        """
        Splits _text_ in token groups that represent words. Splitting is
        performed along whitespaces, so a preformatted text is required.

        :param text: the input text.
        :return: word, token id tensor pairs.
        """
        words = text.split()
        ret = []
        for wid, word in enumerate(words):
            eff_word = f' {word.lower()}' if wid else word.lower()
            tokens = self.tokenize(eff_word)
            ret.append((word, tokens))
        return ret

    def distractor_probabilities(self, prompt, candidates) -> torch.Tensor:
        """
        Computes the log probabilities of all _candidates_ after _prompt_.
        Splits up _candidates_ into chunks that fit the batch size
        specified in :meth:`__init__`.

        :return: the log probabilities.
        """
        logging.debug(f'XXXX {prompt=} {candidates=} ({len(candidates)})')
        x = torch.concat([
            self._distractor_probabilities_batch(prompt, candidates_batch).cpu()
            for candidates_batch in chunked(candidates, self.batch_size)
        ])
        logging.debug(f'YYYY {x=}')
        return x

    def _distractor_probabilities_batch(self, prompt, candidates):
        """
        Computes and returns the log probabilities of all _candidates_ after
        _prompt_.

        The candidates are tokenized and the probabilities are computed
        parallelly in a single batch.
        """
        logging.debug(f'{prompt["input_ids"].size(1)}')
        candidates_tokenized = self.tokenize(candidates)
        model_input = {
            k: torch.concat([prompt[k].repeat(len(candidates), 1), v], 1)
            for k, v in candidates_tokenized.items()
        }
        logging.debug(f'{model_input=}')
        with torch.no_grad():
            logits = self.model(**model_input).logits
        # -1, because the logits for the first character of the distractor are
        # above the last prompt token
        log_probs = torch.log_softmax(
            logits, 2
        )[:, prompt['input_ids'].size(1) - 1: -1]
        logging.debug(f'{log_probs[0]=}')
        # We don't care about the probability of EOS tokens and the last
        # non-EOS token, either. Note that the window is shifted 1 to the left,
        # so we can use the attention mask for probability masking as well.
        log_probs[candidates_tokenized['attention_mask'] == 0] = 0
        logging.debug(f'{log_probs[0]=}')
        # Index
        log_probs2 = torch.gather(
            log_probs, 2, candidates_tokenized['input_ids'].unsqueeze(2)
        )
        logging.debug(f'{log_probs2[0]=}')
        logging.debug(f'probs: {log_probs2.sum(1)}')
        word_log_probs = log_probs2.sum(1).squeeze(1)
        logging.debug(f'Word log probs: min: {word_log_probs.min()} '
                      f'mean={word_log_probs.mean()} '
                      f'max={word_log_probs.max()}')
        return word_log_probs

    def probabilities(self, prompt):
        inputs = self.tokenizer([prompt],
                                return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs, max_new_tokens=1,
            return_dict_in_generate=True, output_scores=True
        )
        logps = torch.log(torch.softmax(outputs.scores[0][0]))
        return logps


class RandomProbabilityModel(ProbabilityModel):
    def device(self):
        """Always "runs" on the CPU."""
        return torch.device('cpu')

    def distractor_probabilities(self, prompt, candidates) -> torch.Tensor:
        """
        Computes the log probabilities of all _candidates_ after _prompt_.
        Splits up _candidates_ into chunks that fit the batch size
        specified in :meth:`__init__`.

        :return: the log probabilities.
        """
        return torch.tensor([-100] * len(candidates))

    def split_text_into_token_groups(
        self, text
    ) -> list[tuple[str, torch.Tensor]]:
        """
        Splits _text_ in token groups that represent words. Of course, since
        we don't have a model, each word will be its own token. Splitting is
        performed along whitespaces, so a preformatted text is required.

        :param text: the input text.
        :return: word, token id tensor pairs.
        """
        tokens = {'input_ids': torch.tensor([[1]]),
                  'attention_mask': torch.tensor([[1]])}
        return [(word, tokens) for word in text.split()]


def convert_token_to_hun(token: str) -> str:
    """
    Converts _token_, returned by PULI-GPT-2's travesty of a tokenizer to
    proper Python string instead of UTF-8 bytes masquarading as string.

    Although it doesn't work as the tokenizer **splits up UTF-8 characters**.
    """
    cs = [ord(c) for c in token]
    if cs[0] >= 256:
        prefix, cs = token[0], cs[1:]
    else:
        prefix = ''
    return prefix + bytes(cs).decode('utf-8')
