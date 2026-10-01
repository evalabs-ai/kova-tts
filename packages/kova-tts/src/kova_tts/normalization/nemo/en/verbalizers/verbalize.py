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

from ..graph_utils import GraphFst
from .abbreviation import AbbreviationFst
from .cardinal import CardinalFst
from .date import DateFst
from .decimal import DecimalFst
from .electronic import ElectronicFst
from .fraction import FractionFst
from .measure import MeasureFst
from .money import MoneyFst
from .ordinal import OrdinalFst
from .roman import RomanFst
from .telephone import TelephoneFst
from .time import TimeFst
from .whitelist import WhiteListFst


class VerbalizeFst(GraphFst):
    """
    Composes other verbalizer grammars.
    For deployment, this grammar will be compiled and exported to OpenFst Finite State Archive (FAR) File.
    More details to deployment at NeMo/tools/text_processing_deployment.

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="verbalize", kind="verbalize", deterministic=deterministic)
        cardinal = CardinalFst(deterministic=deterministic)
        cardinal_graph = cardinal.fst
        decimal = DecimalFst(cardinal=cardinal, deterministic=deterministic)
        decimal_graph = decimal.fst
        ordinal = OrdinalFst(deterministic=deterministic)
        ordinal_graph = ordinal.fst
        fraction = FractionFst(deterministic=deterministic)
        fraction_graph = fraction.fst
        telephone_graph = TelephoneFst(deterministic=deterministic).fst
        electronic_graph = ElectronicFst(deterministic=deterministic).fst
        measure = MeasureFst(decimal=decimal, cardinal=cardinal, fraction=fraction, deterministic=deterministic)
        measure_graph = measure.fst
        time_graph = TimeFst(deterministic=deterministic).fst
        date_graph = DateFst(ordinal=ordinal, deterministic=deterministic).fst
        money_graph = MoneyFst(decimal=decimal, deterministic=deterministic).fst
        whitelist_graph = WhiteListFst(deterministic=deterministic).fst

        graph = (
            time_graph
            | date_graph
            | money_graph
            | measure_graph
            | ordinal_graph
            | decimal_graph
            | cardinal_graph
            | telephone_graph
            | electronic_graph
            | fraction_graph
            | whitelist_graph
        )

        roman_graph = RomanFst(deterministic=deterministic).fst
        graph |= roman_graph

        if not deterministic:
            abbreviation_graph = AbbreviationFst(deterministic=deterministic).fst
            graph |= abbreviation_graph

        self.fst = graph
