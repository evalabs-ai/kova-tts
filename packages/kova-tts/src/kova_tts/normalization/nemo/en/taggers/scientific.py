"""Scientific notation spoken literally: 1e-3 -> one e negative three."""
import pynini
from pynini.lib import pynutil

from ..graph_utils import GraphFst, convert_space


class ScientificFst(GraphFst):
    def __init__(self, cardinal, deterministic=True):
        super().__init__(name="scientific", kind="classify", deterministic=deterministic)
        # Strip exponent/quantity padding without floating point conversion.
        integer = (pynini.closure(pynutil.delete("0")) + cardinal.graph).optimize()
        fraction = pynini.cross(".", " point ") + cardinal.single_digits_graph
        mantissa = integer + fraction.ques
        mantissa |= pynutil.insert("zero") + fraction
        sign = pynini.string_map([("-", "negative "), ("−", "negative "), ("+", "plus ")]).ques
        graph = sign + mantissa + pynini.cross(pynini.union("e", "E"), " e ") + sign + integer
        self.graph = graph.optimize()
        # A name token needs no additional protobuf class or verbalizer.
        self.fst = (pynutil.insert('name: "') + convert_space(graph) + pynutil.insert('"')).optimize()
