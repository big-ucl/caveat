import torch

from caveat.encoding import BaseDataset, PaddedDatatset


def test_base_encoded():
    encoded = BaseDataset(
        schedules=torch.rand((3, 12)),
        act_weights=torch.ones((3, 12)),
        seq_weights=torch.ones((3, 1)),
        activity_encodings=4,
        activity_weights=torch.ones(4),
        augment=None,
        labels=torch.ones((3, 3)),
        label_weights=None,
        joint_weights=None,
    )
    for i in range(len(encoded)):
        (left, (left_mask, _)), (right, (right_mask, _)), (labels, _) = encoded[
            i
        ]
        assert left.shape == (12,)
        assert left_mask.shape == (12,)
        assert right.shape == (12,)
        assert right_mask.shape == (12,)
        assert labels.shape == (3,)


def test_base_encoded_padded():
    encoded = PaddedDatatset(
        schedules=torch.rand((3, 12)),
        act_weights=torch.ones((3, 13)),
        seq_weights=torch.ones((3, 1)),
        activity_encodings=4,
        activity_weights=torch.ones(4),
        augment=None,
        labels=torch.ones((3, 3)),
        label_weights=None,
        joint_weights=None,
    )
    for i in range(len(encoded)):
        (left, (left_mask, _)), (right, (right_mask, _)), (labels, _) = encoded[
            i
        ]
        assert left.shape == (13,)
        assert left_mask.shape == (13,)
        assert right.shape == (13,)
        assert right_mask.shape == (13,)
        assert labels.shape == (3,)
