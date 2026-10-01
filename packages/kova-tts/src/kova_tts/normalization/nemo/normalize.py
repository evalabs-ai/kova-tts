# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Trimmed for kova-tts: the English inference path only. The other languages, the batch and
# manifest helpers and the command-line entry point are gone, and with them joblib and tqdm.
# Everything kept is unchanged, except that normalize() no longer resets the log level to INFO
# on every call (see logging.py).

import itertools
import re
import sys
from collections import OrderedDict
from math import factorial
from typing import Dict, List, Union

import pynini
from pynini.lib.rewrite import top_rewrite
from sacremoses import MosesDetokenizer

from .data_loader_utils import post_process_punct, pre_process
from .logging import logger
from .token_parser import PRESERVE_ORDER_KEY, TokenParser

# this is to handle long input
sys.setrecursionlimit(3000)

SPACE_DUP = re.compile(' {2,}')


class Normalizer:
    """
    Normalizer class that converts text from written to spoken form.
    Useful for TTS preprocessing.

    Args:
        input_case: Input text capitalization, set to 'cased' if text contains capital letters.
            This flag affects normalization rules applied to the text. Note, `lower_cased` won't lower case input.
        lang: language specifying the TN rules; only English is vendored here
        cache_dir: path to a dir with .far grammar file. Set to None to avoid using cache.
        overwrite_cache: set to True to overwrite .far files
        whitelist: path to a file with whitelist replacements
        post_process: WFST-based post processing, e.g. to remove extra spaces added during TN.
            Note: punct_post_process flag in normalize() supports all languages.
        max_number_of_permutations_per_split: a maximum number
            of permutations which can be generated from input sequence of tokens.
    """

    def __init__(
        self,
        input_case: str,
        lang: str = 'en',
        deterministic: bool = True,
        cache_dir: str = None,
        overwrite_cache: bool = False,
        whitelist: str = None,
        post_process: bool = True,
        max_number_of_permutations_per_split: int = 729,
    ):
        assert input_case in ["lower_cased", "cased"]
        if lang != "en" or not deterministic:
            raise NotImplementedError("Only deterministic English normalization is vendored.")

        from .en.taggers.tokenize_and_classify import ClassifyFst
        from .en.verbalizers.post_processing import PostProcessingFst
        from .en.verbalizers.verbalize_final import VerbalizeFinalFst

        self.post_processor = None
        if post_process:
            self.post_processor = PostProcessingFst(cache_dir=cache_dir, overwrite_cache=overwrite_cache)

        self.input_case = input_case
        self.tagger = ClassifyFst(
            input_case=self.input_case,
            deterministic=deterministic,
            cache_dir=cache_dir,
            overwrite_cache=overwrite_cache,
            whitelist=whitelist,
        )

        self.verbalizer = VerbalizeFinalFst(
            deterministic=deterministic, cache_dir=cache_dir, overwrite_cache=overwrite_cache
        )
        self.max_number_of_permutations_per_split = max_number_of_permutations_per_split
        self.parser = TokenParser()
        self.lang = lang
        self.moses_detokenizer = MosesDetokenizer(lang=lang)

    def _estimate_number_of_permutations_in_nested_dict(
        self, token_group: Dict[str, Union[OrderedDict, str, bool]]
    ) -> int:
        num_perms = 1
        for k, inner in token_group.items():
            if isinstance(inner, dict):
                num_perms *= self._estimate_number_of_permutations_in_nested_dict(inner)
        num_perms *= factorial(len(token_group))
        return num_perms

    def _split_tokens_to_reduce_number_of_permutations(self, tokens: List[dict]) -> List[List[dict]]:
        """
        Splits a sequence of tokens in a smaller sequences of tokens in a way that maximum number of composite
        tokens permutations does not exceed ``max_number_of_permutations_per_split``.

        For example,

        .. code-block:: python

            # setup normalizer with self.max_number_of_permutations_per_split=6
             tokens = [
                {"tokens": {"date": {"year": "twenty eighteen", "month": "december", "day": "thirty one"}}},
                {"tokens": {"date": {"year": "twenty eighteen", "month": "january", "day": "eight"}}},
            ]
            split = normalizer._split_tokens_to_reduce_number_of_permutations(tokens)
            assert split == [
                [{"tokens": {"date": {"year": "twenty eighteen", "month": "december", "day": "thirty one"}}}],
                [{"tokens": {"date": {"year": "twenty eighteen", "month": "january", "day": "eight"}}}],
            ]

        Date tokens contain 3 items each which gives 6 permutations for every date. Since there are 2 dates, total
        number of permutations would be ``6 * 6 == 36``. Parameter ``self.max_number_of_permutations_per_split`` equals 6,
        so input sequence of tokens is split into 2 smaller sequences.

        Args:
            tokens: a list of dictionaries, possibly nested.

        Returns:
            a list of smaller sequences of tokens resulting from ``tokens`` split.
        """
        splits = []
        prev_end_of_split = 0
        current_number_of_permutations = 1
        for i, token_group in enumerate(tokens):
            n = self._estimate_number_of_permutations_in_nested_dict(token_group)
            if n * current_number_of_permutations > self.max_number_of_permutations_per_split:
                splits.append(tokens[prev_end_of_split:i])
                prev_end_of_split = i
                current_number_of_permutations = 1
            if n > self.max_number_of_permutations_per_split:
                raise ValueError(
                    f"Could not split token list with respect to condition that every split can generate number of "
                    f"permutations less or equal to "
                    f"`self.max_number_of_permutations_per_split={self.max_number_of_permutations_per_split}`. "
                    f"There is an unsplittable token group that generates more than "
                    f"{self.max_number_of_permutations_per_split} permutations. Try to increase "
                    f"`--max_number_of_permutations_per_split` parameter."
                )
            current_number_of_permutations *= n
        splits.append(tokens[prev_end_of_split:])
        assert sum([len(s) for s in splits]) == len(tokens)
        return splits

    def normalize(
        self, text: str, verbose: bool = False, punct_pre_process: bool = False, punct_post_process: bool = False
    ) -> str:
        """
        Main function. Normalizes tokens from written to spoken form
            e.g. 12 kg -> twelve kilograms

        Args:
            text: string that may include semiotic classes
            punct_pre_process: whether to perform punctuation pre-processing, for example, [25] -> [ 25 ]
            punct_post_process: whether to normalize punctuation
            verbose: whether to print intermediate meta information

        Returns: spoken form
        """
        if len(text.split()) > 500:
            logger.warning(
                "Your input is too long and could take a long time to normalize. "
                "Use split_text_into_sentences() to make the input shorter and then call normalize_list()."
            )
        original_text = text
        if punct_pre_process:
            text = pre_process(text)
        text = text.strip()
        if not text:
            logger.debug(text)
            return text
        text = pynini.escape(text)
        tagged_lattice = self.find_tags(text)
        tagged_text = Normalizer.select_tag(tagged_lattice)
        logger.debug(tagged_text)

        self.parser(tagged_text)
        tokens = self.parser.parse()
        split_tokens = self._split_tokens_to_reduce_number_of_permutations(tokens)
        output = ""
        for s in split_tokens:
            try:
                tags_reordered = self.generate_permutations(s)
                verbalizer_lattice = None
                for tagged_text in tags_reordered:
                    tagged_text = pynini.escape(tagged_text)

                    verbalizer_lattice = self.find_verbalizer(tagged_text)
                    if verbalizer_lattice.num_states() != 0:
                        break
                if verbalizer_lattice is None:
                    logger.warning(f"No permutations were generated from tokens {s}")
                    return text
                output += ' ' + Normalizer.select_verbalizer(verbalizer_lattice)
            except Exception as e:
                logger.warning("Failed text: " + text + str(e))
                return text
        output = SPACE_DUP.sub(' ', output[1:])

        if self.lang == "en" and hasattr(self, 'post_processor'):
            output = self.post_process(output)

        if punct_post_process:
            # do post-processing based on Moses detokenizer
            output = self.moses_detokenizer.detokenize([output], unescape=False)
            output = post_process_punct(input=original_text, normalized_text=output)
        return output

    def _permute(self, d: OrderedDict) -> List[str]:
        """
        Creates reorderings of dictionary elements and serializes as strings

        Args:
            d: (nested) dictionary of key value pairs

        Return permutations of different string serializations of key value pairs
        """
        l = []
        if PRESERVE_ORDER_KEY in d.keys():
            d_permutations = [d.items()]
        else:
            d_permutations = itertools.permutations(d.items())
        for perm in d_permutations:
            subl = [""]
            for k, v in perm:
                if isinstance(v, str):
                    subl = ["".join(x) for x in itertools.product(subl, [f"{k}: \"{v}\" "])]
                elif isinstance(v, OrderedDict):
                    rec = self._permute(v)
                    subl = ["".join(x) for x in itertools.product(subl, [f" {k} {{ "], rec, [f" }} "])]
                elif isinstance(v, bool):
                    subl = ["".join(x) for x in itertools.product(subl, [f"{k}: true "])]
                else:
                    raise ValueError("Key: " + str(k) + " Value: " + str(v))
            l.extend(subl)
        return l

    def generate_permutations(self, tokens: List[dict]):
        """
        Generates permutations of string serializations of list of dictionaries

        Args:
            tokens: list of dictionaries

        Returns string serialization of list of dictionaries
        """

        def _helper(prefix: str, token_list: List[dict], idx: int):
            """
            Generates permutations of string serializations of given dictionary

            Args:
                token_list: list of dictionaries
                prefix: prefix string
                idx:    index of next dictionary

            Returns string serialization of dictionary
            """
            if idx == len(token_list):
                yield prefix
                return
            token_options = self._permute(token_list[idx])
            for token_option in token_options:
                yield from _helper(prefix + token_option, token_list, idx + 1)

        return _helper("", tokens, 0)

    def find_tags(self, text: str) -> 'pynini.FstLike':
        """
        Given text use tagger Fst to tag text

        Args:
            text: sentence

        Returns: tagged lattice
        """
        lattice = text @ self.tagger.fst
        return lattice

    @staticmethod
    def select_tag(lattice: 'pynini.FstLike') -> str:
        """
        Given tagged lattice return shortest path

        Args:
            lattice: pynini.FstLike tag lattice

        Returns: shortest path
        """
        tagged_text = pynini.shortestpath(lattice, nshortest=1, unique=True).string()
        return tagged_text

    def find_verbalizer(self, tagged_text: str) -> 'pynini.FstLike':
        """
        Given tagged text creates verbalization lattice
        This is context-independent.

        Args:
            tagged_text: input text

        Returns: verbalized lattice
        """
        lattice = tagged_text @ self.verbalizer.fst
        return lattice

    @staticmethod
    def select_verbalizer(lattice: 'pynini.FstLike') -> str:
        """
        Given verbalized lattice return shortest path

        Args:
            lattice: verbalization lattice
            text: full text line to raise in case of an exception

        Returns: shortest path
        """
        output = pynini.shortestpath(lattice, nshortest=1, unique=True).string()
        # lattice = output @ self.verbalizer.punct_graph
        # output = pynini.shortestpath(lattice, nshortest=1, unique=True).string()
        return output

    def post_process(self, normalized_text: 'pynini.FstLike') -> str:
        """
        Runs post-processing graph on normalized text

        Args:
            normalized_text: normalized text

        Returns: shortest path
        """
        normalized_text = normalized_text.strip()
        if not normalized_text:
            return normalized_text
        normalized_text = pynini.escape(normalized_text)

        if self.post_processor is not None:
            normalized_text = top_rewrite(normalized_text, self.post_processor.fst)
        return normalized_text
