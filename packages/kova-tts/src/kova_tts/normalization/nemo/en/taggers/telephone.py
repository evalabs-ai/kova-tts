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

import pynini
from pynini.lib import pynutil

from ..graph_utils import (
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_SIGMA,
    GraphFst,
    delete_extra_space,
    NEMO_UPPER,
    delete_space,
    insert_space,
    plurals,
)
from ..utils import get_abs_path, load_labels


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying telephone, and IP, and SSN which includes country code, number part and extension
    Explicit phone layouts preserve digit groups with commas.
    Country prefixes and labeled extensions retain every digit.
    E.g
    123-456-7890 -> telephone { number_part: "one two three, four five six, seven eight nine zero" }
    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="telephone", kind="classify", deterministic=deterministic)

        add_separator = pynutil.insert(", ")  # between components
        zero = pynini.cross("0", "zero")
        if not deterministic:
            zero |= pynini.cross("0", pynini.union("o", "oh"))
        digit = pynini.invert(pynini.string_file(get_abs_path("data/number/digit.tsv"))).optimize() | zero

        def group(minimum, maximum=None):
            return digit + (insert_space + digit) ** (minimum - 1, (maximum or minimum) - 1)

        # Preserve each digit (including 800 and leading zeros). Formatting,
        # rather than NANP validity, identifies a phone: synthetic numbers are
        # useful model input too. Require consistent separators in plain forms.
        three, four = group(3), group(4)
        compact = three + add_separator + three + add_separator + four
        national = pynini.union(*(
            three + pynini.cross(separator, ", ") + three
            + pynini.cross(separator, ", ") + four
            for separator in ("-", ".", " ", "‐", "‑", "–")
        ))
        area = (pynutil.delete("(") + three + pynutil.delete(")")
                + pynini.closure(pynutil.delete(pynini.union(" ", "-")), 0, 1) + add_separator)
        subscriber = three + pynini.cross(pynini.union("-", ".", " "), ", ") + four
        national |= area + (subscriber | (three + add_separator + four))
        local = three + pynini.cross("-", ", ") + four
        # A lone 555.0123 is also a decimal. Only a phone cue can resolve it.
        prompted_local = three + pynini.cross(".", ", ") + four

        # Vanity phones retain their literal letters, spoken individually.
        # Only a three-digit area and exactly seven subscriber symbols qualify;
        # ordinary all-caps words outside a phone are never spelled by this rule.
        letter = NEMO_UPPER | pynini.string_map([(chr(i), chr(i).upper()) for i in range(97, 123)])
        symbol = digit | letter
        vanity = symbol + ((insert_space | pynini.cross("-", ", ")) + symbol) ** 6
        vanity = (NEMO_SIGMA + NEMO_ALPHA + NEMO_SIGMA) @ vanity
        national |= (three + pynini.cross(pynini.union("-", ".", " "), ", ") | area) + vanity
        national = national.optimize()

        country_separator = pynutil.delete(pynini.union(" ", "-", "."))
        north_american = (
            pynini.cross("1", "one, ") + country_separator + national
            | pynini.cross("+1", "plus one, ")
              + ((country_separator + (national | compact)) | compact)
        )
        # International numbers must have an explicit '+' and separated groups.
        # Preserve the supplied grouping rather than guessing a national plan.
        country = (pynini.difference(NEMO_DIGIT, "0") @ digit) + (insert_space + digit) ** (0, 2)
        international = (pynini.cross("+", "plus ") + country
                         + country_separator + add_separator
                         + group(1, 8) + (pynini.cross(pynini.union(" ", "-", "."), ", ") + group(1, 8)) ** (1, 5))
        # Bound complete international forms to 7..15 digits (country included).
        punctuation = pynini.closure(pynini.union(" ", "-", ".", "+"))
        length = punctuation + (NEMO_DIGIT + punctuation) ** (7, 15)
        international = length @ international
        number = plurals._priority_union(north_american, international, NEMO_SIGMA)
        number |= national | local

        # Unformatted ten-digit quantities stay on the existing number path.
        # Explicit phone cues authorize 3-3-4 grouping without changing prose.
        prompts = [row[0] for row in load_labels(get_abs_path("data/telephone/telephone_prompt.tsv"))]
        prompts += ["call", "call me", "phone", "phone number", "telephone", "tel", "mobile", "fax"]
        cues = pynini.string_map([(form + ending, form + ending) for prompt in prompts
                                  for form in (prompt, prompt.capitalize(), prompt.upper())
                                  for ending in (" ", ": ")])
        number |= cues + (number | compact | prompted_local)

        # Extensions are digit sequences, not quantities: ext. 0042 preserves
        # both zeros. Consume the label with its number so it is not read 'ext'.
        label = pynini.union("ext", "Ext", "EXT", "extension", "Extension", "EXTENSION", "x", "X", "#")
        extension = (delete_space + pynutil.delete(label)
                     + pynini.closure(pynutil.delete("."), 0, 1) + delete_space
                     + pynutil.insert(", extension ") + group(1, 8))
        number += pynini.closure(extension, 0, 1)
        graph = pynutil.insert('number_part: "') + number + pynutil.insert('"')

        # ip
        ip_prompts = pynini.string_file(get_abs_path("data/telephone/ip_prompt.tsv"))
        digit_to_str_graph = digit + pynini.closure(pynutil.insert(" ") + digit, 0, 2)
        ip_graph = digit_to_str_graph + (pynini.cross(".", " dot ") + digit_to_str_graph) ** 3
        graph |= (
            pynini.closure(
                pynutil.insert("country_code: \"") + ip_prompts + pynutil.insert("\"") + delete_extra_space, 0, 1
            )
            + pynutil.insert("number_part: \"")
            + ip_graph.optimize()
            + pynutil.insert("\"")
        )
        # ssn
        ssn_prompts = pynini.string_file(get_abs_path("data/telephone/ssn_prompt.tsv"))
        three_digit_part = digit + (pynutil.insert(" ") + digit) ** 2
        two_digit_part = digit + pynutil.insert(" ") + digit
        four_digit_part = digit + (pynutil.insert(" ") + digit) ** 3
        ssn_separator = pynini.cross("-", ", ")
        ssn_graph = three_digit_part + ssn_separator + two_digit_part + ssn_separator + four_digit_part

        graph |= (
            pynini.closure(
                pynutil.insert("country_code: \"") + ssn_prompts + pynutil.insert("\"") + delete_extra_space, 0, 1
            )
            + pynutil.insert("number_part: \"")
            + ssn_graph.optimize()
            + pynutil.insert("\"")
        )

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
