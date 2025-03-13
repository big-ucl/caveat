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
from caveat.encoding.sequence import (
    ContinuousEncoder,
    ContinuousEncoderStaggered,
)

library = {
    "discrete": DiscreteEncoder,
    "discrete_padded": DiscreteEncoderPadded,
    "continuous": ContinuousEncoder,
    "continous_staggered": ContinuousEncoderStaggered,
    "sequence": ContinuousEncoder,
    "sequence_staggered": ContinuousEncoderStaggered,
    "seq2seq": Seq2SeqEncoder,
    "seq2score": Seq2ScoreEncoder,
}
