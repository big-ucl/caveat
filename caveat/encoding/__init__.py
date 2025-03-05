from caveat.encoding.base import (
    BaseDataset,
    BaseEncoder,
    LHS2RHSDataset,
    PaddedDatatset,
    StaggeredDataset,
)
from caveat.encoding.discrete import DiscreteEncoder, DiscreteEncoderPadded
from caveat.encoding.seq2score import Seq2ScoreEncoder
from caveat.encoding.seq2seq import Seq2SeqEncoder
from caveat.encoding.seq_weighting import act_weight_library, seq_weight_library
from caveat.encoding.sequence import SequenceEncoder, SequenceEncoderStaggered

library = {
    "discrete": DiscreteEncoder,
    "discrete_padded": DiscreteEncoderPadded,
    "sequence": SequenceEncoder,
    "sequence_staggered": SequenceEncoderStaggered,
    "seq2seq": Seq2SeqEncoder,
    "seq2score": Seq2ScoreEncoder,
}
