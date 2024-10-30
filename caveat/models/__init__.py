from .base import Base
from .discrete.auto_discrete_lstm import AutoDiscLSTM
from .discrete.cond_discrete_conv import CondDiscConv
from .discrete.cond_discrete_lstm import CondDiscLSTM
from .discrete.vae_discrete_conv import VAEDiscConv
from .discrete.vae_discrete_lstm import VAEDiscLSTM
from .discrete.vae_discrete_transformer import VAEDiscTrans
from .embed import CustomDurationEmbedding, CustomDurationModeDistanceEmbedding
from .joint_vaes.jvae_sequence import JVAESeqLSTM
from .joint_vaes.jvae_sequence_rerouted import JVAESeqLSTMRerouted
from .seq2score.lstm import Seq2ScoreLSTM
from .seq2seq.lstm import Seq2SeqLSTM
from .sequence.auto_sequence_lstm import AutoSeqLSTM
from .sequence.cond_sequence_lstm import CondSeqLSTM
from .sequence.cvae_sequence_lstm import CVAESeqLSTM
from .sequence.cvae_sequence_lstm_add import CVAESeqLSTMAdd
from .sequence.cvae_sequence_lstm_after import CVAESeqLSTMAfter
from .sequence.cvae_sequence_lstm_all import CVAESeqLSTMAll
from .sequence.cvae_sequence_lstm_double_nudger import CVAESeqLSTMDoubleNudger
from .sequence.cvae_sequence_lstm_feed import CVAESeqLSTMFeed
from .sequence.cvae_sequence_lstm_nudger import CVAESeqLSTMNudger
from .sequence.cvae_sequence_lstm_nudger_adversarial import (
    CVAESeqLSTMNudgerAdversarial,
)
from .sequence.vae_sequence_lstm import VAESeqLSTM

library = {
    "CondDiscLSTM": CondDiscLSTM,  # lstm unit given attributes as input at every step
    "CondDiscConv": CondDiscConv,  # similar to koushik but with CNN
    "CondSeqLSTM": CondSeqLSTM,  # lstm unit given attributes as input at first step
    "AutoDiscLSTM": AutoDiscLSTM,  # lstm unit input is previous output
    "AutoSeqLSTM": AutoSeqLSTM,
    "VAEDiscConv": VAEDiscConv,
    "VAESeqLSTM": VAESeqLSTM,
    "VAEDiscLSTM": VAEDiscLSTM,
    "VAEDiscTrans": VAEDiscTrans,
    "CVAESeqLSTM": CVAESeqLSTM,  # attributes at decoder only
    "CVAESeqLSTMAdd": CVAESeqLSTMAdd,  # adds conditionals to latent layer
    "CVAESeqLSTMFeed": CVAESeqLSTMFeed,  # passes conditionals to decoder units
    "CVAESeqLSTMAfter": CVAESeqLSTMAfter,  # conditionals concat after LSTM output
    "CVAESeqLSTMAll": CVAESeqLSTMAll,  # labels cat to latents and added to unit in/outs
    "CVAESeqLSTMNudge": CVAESeqLSTMNudger,  # nudger model
    "CVAESeqLSTMDoubleNudge": CVAESeqLSTMDoubleNudger,  # double nudger model
    "CVAESeqLSTMNudgeAdv": CVAESeqLSTMNudgerAdversarial,  # adversarial nudger model
    "Seq2SeqLSTM": Seq2SeqLSTM,
    "Seq2ScoreLSTM": Seq2ScoreLSTM,
    "JVAESeqLSTM": JVAESeqLSTM,
    "JVAESeqLSTMRerouted": JVAESeqLSTMRerouted,
}
