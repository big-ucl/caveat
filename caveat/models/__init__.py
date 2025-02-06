from .base import Base
from .discrete.auto_discrete_lstm import AutoDiscLSTM
from .discrete.cond_discrete_conv import CondDiscCNN2D
from .discrete.cond_discrete_lstm import CondDiscLSTM
from .discrete.vae_discrete_cnn1d import VAEDiscCNN1D
from .discrete.vae_discrete_cnn2d import VAEDiscCNN2D
from .discrete.vae_discrete_fc import VAEDiscFC
from .discrete.vae_discrete_lstm import VAEDiscLSTM
from .discrete.vae_discrete_xattention import VAEDiscXTrans
from .embed import (
    CustomDurationEmbeddingAddNorm,
    CustomDurationEmbeddingConcat,
    CustomDurationModeDistanceEmbedding,
)
from .joint_vaes.jvae_sequence import JVAESeqLSTM
from .joint_vaes.jvae_sequence_rerouted import JVAESeqLSTMRerouted
from .schedule2label.feedforward import Schedule2LabelFeedForward
from .seq2score.lstm import Seq2ScoreLSTM
from .seq2seq.lstm import Seq2SeqLSTM
from .sequence.auto_sequence_attention import AutoSeqAtt
from .sequence.auto_sequence_lstm import AutoSeqLSTM
from .sequence.cond_sequence_lstm import CondSeqLSTM
from .sequence.cvae_sequence_lstm import CVAESeqLSTM
from .sequence.cvae_sequence_lstm_double_nudger import CVAESeqLSTMDoubleNudger
from .sequence.cvae_sequence_lstm_nudge_feed import (
    CVAESeqLSTMNudgeFeed,
    CVAESeqLSTMNudgeFeedPre,
)
from .sequence.cvae_sequence_lstm_nudger import CVAESeqLSTMNudger
from .sequence.cvae_sequence_lstm_nudger_adversarial import (
    CVAESeqLSTMNudgerAdversarial,
)
from .sequence.vae_sequence_attention import VAESeqXAtt
from .sequence.vae_sequence_cnn import VAESeqCNN2D
from .sequence.vae_sequence_cnn1d import VAESeqCNN1D
from .sequence.vae_sequence_fc import VAESeqFC
from .sequence.vae_sequence_lstm import VAESeqLSTM

library = {
    "CondDiscLSTM": CondDiscLSTM,  # lstm unit given attributes as input at every step
    "CondDiscCNN2D": CondDiscCNN2D,  # similar to koushik but with CNN
    "CondSeqLSTM": CondSeqLSTM,  # lstm unit given attributes as input at first step
    "AutoDiscLSTM": AutoDiscLSTM,  # lstm unit input is previous output
    "AutoSeqLSTM": AutoSeqLSTM,
    "AutoSeqAtt": AutoSeqAtt,
    "VAEDiscCNN2D": VAEDiscCNN2D,
    "VAEDiscCNN1D": VAEDiscCNN1D,
    "VAEDiscFC": VAEDiscFC,
    "VAEDiscLSTM": VAEDiscLSTM,
    "VAEDiscTrans": VAEDiscXTrans,
    "VAESeqLSTM": VAESeqLSTM,
    "VAESeqCNN2D": VAESeqCNN2D,
    "VAESeqCNN1D": VAESeqCNN1D,
    "VAESeqFC": VAESeqFC,
    "VAESeqXAtt": VAESeqXAtt,
    "CVAESeqLSTM": CVAESeqLSTM,  # attributes at decoder only
    "CVAESeqLSTMNudgeFeed": CVAESeqLSTMNudgeFeed,  # nudger model
    "CVAESeqLSTMNudgeFeedPre": CVAESeqLSTMNudgeFeedPre,  # nudger model
    "CVAESeqLSTMNudge": CVAESeqLSTMNudger,  # nudger model
    "CVAESeqLSTMDoubleNudge": CVAESeqLSTMDoubleNudger,  # double nudger model
    "CVAESeqLSTMNudgeAdv": CVAESeqLSTMNudgerAdversarial,  # adversarial nudger model
    "Seq2SeqLSTM": Seq2SeqLSTM,
    "Seq2ScoreLSTM": Seq2ScoreLSTM,
    "JVAESeqLSTM": JVAESeqLSTM,
    "JVAESeqLSTMRerouted": JVAESeqLSTMRerouted,
    "LabelFeedForward": Schedule2LabelFeedForward,
}
