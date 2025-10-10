from .base import Base
from .cat_base import CatBase
from .continuous.auto_attention import AutoContAtt
from .continuous.auto_lstm import AutoContLSTM
from .continuous.cat_vae_lstm import CatVAEContLSTM
from .continuous.cond_lstm import CondContLSTM
from .continuous.cswae_lstm import CSWAEContLSTM
from .continuous.cvae_lstm import CVAEContLSTM
from .continuous.cvae_lstm_countdown import CVAEContLSTMCountdown
from .continuous.cvqvae_lstm import CVQVAEContLSTM
from .continuous.swae_lstm import SWAEContLSTM
from .continuous.vae_attention import VAEContXAtt
from .continuous.vae_cnn1d import VAEContCNN1D
from .continuous.vae_cnn2d import VAEContCNN2D
from .continuous.vae_fc import VAEContFC
from .continuous.vae_lstm import VAEContLSTM
from .continuous.vae_lstm_countdown import VAEContLSTMCountdown
from .continuous.vqvae_lstm import VQVAEContLSTM
from .discrete.auto_discrete_lstm import AutoDiscLSTM
from .discrete.cond_discrete_conv import CondDiscCNN2D
from .discrete.cond_discrete_lstm import CondDiscLSTM
from .discrete.vae_discrete_cnn1d import VAEDiscCNN1D
from .discrete.vae_discrete_cnn2d import VAEDiscCNN2D
from .discrete.vae_discrete_fc import VAEDiscFC
from .discrete.vae_discrete_lstm import VAEDiscLSTM
from .discrete.vae_discrete_xattention import VAEDiscXTrans
from .joint_vaes.jvae_continuous import JVAEContLSTM
from .joint_vaes.jvae_continuous_rerouted import JVAEContLSTMRerouted
from .schedule2label.feedforward import Schedule2LabelFeedForward
from .seq2score.lstm import Seq2ScoreLSTM
from .seq2seq.lstm import Seq2SeqLSTM

library = {
    "CondDiscLSTM": CondDiscLSTM,
    "CondDiscCNN2D": CondDiscCNN2D,
    "CondContLSTM": CondContLSTM,
    "AutoDiscLSTM": AutoDiscLSTM,
    "AutoContLSTM": AutoContLSTM,
    "AutoContAtt": AutoContAtt,
    "VAEDiscCNN2D": VAEDiscCNN2D,
    "VAEDiscCNN1D": VAEDiscCNN1D,
    "VAEDiscFC": VAEDiscFC,
    "VAEDiscLSTM": VAEDiscLSTM,
    "VAEDiscTrans": VAEDiscXTrans,
    "VAEContLSTM": VAEContLSTM,
    "SWAEContLSTM": SWAEContLSTM,
    "VAEContCNN2D": VAEContCNN2D,
    "VAEContCNN1D": VAEContCNN1D,
    "VAEContFC": VAEContFC,
    "VAEContXAtt": VAEContXAtt,
    "CVAEContLSTM": CVAEContLSTM,
    "CSWAEContLSTM": CSWAEContLSTM,
    "Seq2SeqLSTM": Seq2SeqLSTM,
    "Seq2ScoreLSTM": Seq2ScoreLSTM,
    "JVAEContLSTM": JVAEContLSTM,
    "JVAEContLSTMRerouted": JVAEContLSTMRerouted,
    "LabelFeedForward": Schedule2LabelFeedForward,
    "CatVAEContLSTM": CatVAEContLSTM,
    "VAEContLSTMCountdown": VAEContLSTMCountdown,
    "CVAEContLSTMCountdown": CVAEContLSTMCountdown,
    "VQVAEContLSTM": VQVAEContLSTM,
    "CVQVAEContLSTM": CVQVAEContLSTM,
}
