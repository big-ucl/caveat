from abc import ABC
from typing import Optional

from pandas import DataFrame
from torch import Tensor
from torch.nn.functional import pad
from torch.utils.data import Dataset

from caveat.data import ScheduleAugment


class BaseEncoder(ABC):

    def __init__(self, schedules: DataFrame, **kwargs) -> None:
        raise NotImplementedError

    def encode(
        self,
        schedules: DataFrame,
        labels: Optional[Tensor],
        label_weights: Optional[Tensor],
    ) -> Dataset:
        raise NotImplementedError

    def decode(self, schedules: Tensor) -> DataFrame:
        raise NotImplementedError


class BaseDataset(Dataset):
    def __init__(
        self,
        schedules: Tensor,
        act_weights: Optional[Tensor],
        seq_weights: Optional[Tensor],
        activity_encodings: int,
        activity_weights: Optional[Tensor],
        augment: Optional[ScheduleAugment],
        labels: Optional[Tensor],
        label_weights: Optional[Tensor],
        joint_weights: Optional[Tensor],
    ):
        super(BaseDataset, self).__init__()
        self.schedules = schedules
        self.act_weights = act_weights
        self.seq_weights = seq_weights
        self.activity_encodings = activity_encodings
        self.encoding_weights = activity_weights
        self.augment = augment
        self.labels = labels
        self.label_weights = label_weights
        self.joint_weights = joint_weights
        self.labels_shape = labels.shape[-1] if labels is not None else None

    def shape(self):
        return self.schedules[0].shape

    def __len__(self):
        return len(self.schedules)

    def __getitem__(self, idx):
        sample = self.schedules[idx]
        if self.augment:
            sample = self.augment(sample)

        if self.act_weights is not None:
            weights = self.act_weights[idx]
        else:
            weights = None

        if self.seq_weights is not None:
            seq_weights = self.seq_weights[idx]
        else:
            seq_weights = Tensor([])

        if self.labels is not None:
            labels = self.labels[idx]
        else:
            labels = Tensor([])

        if self.label_weights is not None:
            label_weights = self.label_weights[idx]
        else:
            label_weights = Tensor([])

        if self.joint_weights is not None:
            joint_weights = self.joint_weights[idx]
        else:
            joint_weights = Tensor([])

        return (
            (sample, (weights, seq_weights)),
            (sample, (weights, seq_weights)),
            (labels, (label_weights, joint_weights)),
        )


class PaddedDatatset(BaseDataset):

    def shape(self):
        _, L = self.schedules.shape
        return (L + 1,)

    def __getitem__(self, idx):
        sample = self.schedules[idx]
        if self.augment:
            sample = self.augment(sample)

        if self.act_weights is not None:
            weights = self.act_weights[idx]
        else:
            weights = None

        if self.seq_weights is not None:
            seq_weights = self.seq_weights[idx]
        else:
            seq_weights = Tensor([])

        if self.labels is not None:
            label = self.labels[idx]
        else:
            label = Tensor([])

        if self.label_weights is not None:
            label_weights = self.label_weights[idx]
        else:
            label_weights = Tensor([])

        if self.joint_weights is not None:
            joint_weights = self.joint_weights[idx]
        else:
            joint_weights = Tensor([])

        pad_left = pad(sample, (1, 0))
        pad_right = pad(sample, (0, 1))
        return (
            (pad_left, (weights, seq_weights)),
            (pad_right, (weights, seq_weights)),
            (label, (label_weights, joint_weights)),
        )


class StaggeredDataset(BaseDataset):

    def shape(self):
        return len(self.schedules[0]) - 1, 2

    def __getitem__(self, idx):
        sample = self.schedules[idx]
        if self.augment:
            sample = self.augment(sample)

        if self.act_weights is not None:
            weights = self.act_weights[idx]
        else:
            weights = None

        if self.seq_weights is not None:
            seq_weights = self.seq_weights[idx]
        else:
            seq_weights = Tensor([])

        if self.labels is not None:
            label = self.labels[idx]
        else:
            label = Tensor([])

        if self.label_weights is not None:
            label_weights = self.label_weights[idx]
        else:
            label_weights = Tensor([])

        if self.joint_weights is not None:
            joint_weights = self.joint_weights[idx]
        else:
            joint_weights = Tensor([])

        return (
            (sample[:-1, :], (weights[:-1], seq_weights)),
            (sample[1:, :], (weights[1:], seq_weights)),
            (label, (label_weights, joint_weights)),
        )


class LHS2RHSDataset(Dataset):
    def __init__(
        self,
        lhs: Tensor,
        rhs: Tensor,
        lhs_weights: Optional[Tensor],
        rhs_weights: Optional[Tensor],
        act_encodings: int,
        mode_encodings: int,
        activity_weights: Optional[Tensor],
        augment: Optional[ScheduleAugment],
        labels: Optional[Tensor],
    ):
        super(LHS2RHSDataset, self).__init__()
        self.lhs = lhs
        self.rhs = rhs
        self.lhs_weights = lhs_weights
        self.rhs_weights = rhs_weights
        self.activity_encodings = (act_encodings, mode_encodings)
        self.encoding_weights = activity_weights
        self.augment = augment
        self.labels = labels
        self.labels_shape = labels.shape[-1] if labels is not None else None

    def shape(self):
        return self.lhs[0].shape

    def __len__(self):
        return len(self.lhs)

    def __getitem__(self, idx):
        lhs = self.lhs[idx]
        rhs = self.rhs[idx]
        if self.augment:
            lhs = self.augment(lhs)
            rhs = self.augment(rhs)

        if self.lhs_weights is not None:
            lhs_weight = self.lhs_weights[idx]
        else:
            lhs_weight = Tensor([])

        if self.rhs_weights is not None:
            rhs_weight = self.rhs_weights[idx]
        else:
            rhs_weight = Tensor([])

        if self.labels is not None:
            labels = self.labels[idx]
        else:
            labels = Tensor([])

        if self.label_weights is not None:
            label_weight = self.label_weights[idx]
        else:
            label_weight = Tensor([])

        return (lhs, lhs_weight), (rhs, rhs_weight), (labels, label_weight)
