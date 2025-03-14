from caveat.encoding.base import (
    BaseDataset,
    BaseEncoder,
    LHS2RHSDataset,
    PaddedDatatset,
    StaggeredDataset,
)
from caveat.encoding.continuous import (
    ContinuousEncoder,
    ContinuousEncoderStaggered,
)
from caveat.encoding.discrete import DiscreteEncoder, DiscreteEncoderPadded
from caveat.encoding.seq2score import Seq2ScoreEncoder
from caveat.encoding.seq2seq import Seq2SeqEncoder

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
