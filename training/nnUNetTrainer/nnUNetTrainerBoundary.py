from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import DC_and_CE
from nnunetv2.training.loss.deep_supervision import MultipleOutputLoss2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt


class BoundaryLoss(nn.Module):
    """
    Boundary loss for medical image segmentation
    """
    def __init__(self, idc: list):
        super(BoundaryLoss, self).__init__()
        self.idc = idc

    def forward(self, probs: torch.Tensor, dist_maps: torch.Tensor):
        """
        probs: network probabilities (B, C, X, Y, Z)
        dist_maps: precomputed distance maps for ground truth (B, C, X, Y, Z)
        """
        assert probs.shape == dist_maps.shape
        pc = probs[:, self.idc, ...]
        dc = dist_maps[:, self.idc, ...]
        loss = -(pc * dc).mean()  # 取负号
        return loss


def compute_distance_map(mask: np.ndarray) -> np.ndarray:
    """
    mask: numpy array (C, X, Y, Z)
    returns distance map for each class
    """
    dist_map = np.zeros_like(mask, dtype=np.float32)
    for c in range(mask.shape[0]):
        posmask = mask[c].astype(bool)
        if posmask.any():
            negmask = ~posmask
            dist_out = distance_transform_edt(negmask)
            dist_in = distance_transform_edt(posmask)
            dist_map[c] = dist_in - dist_out
    return dist_map


class nnUNetTrainerBoundary(nnUNetTrainer):
    """
    Custom Trainer with DiceLoss + BoundaryLoss
    """

    def _build_loss(self):
        dc_ce_loss = DC_and_CE(
            {}, 
            {'batch_dice': True, 'smooth': 1e-5, 'do_bg': True, 'ddp': self.is_ddp},
            weight_ce=1, weight_dice=1
        )
        boundary_loss = BoundaryLoss(idc=[1])  # 只算 disc

        def combined_loss(output, target):
            # Dice+CE
            loss = dc_ce_loss(output, target)

            # Boundary loss
            with torch.no_grad():
                target_onehot = torch.zeros_like(output)
                target_onehot.scatter_(1, target.long(), 1)

                dist_maps = torch.zeros_like(output)
                for b in range(target_onehot.shape[0]):
                    dist_maps[b] = torch.from_numpy(
                        compute_distance_map(target_onehot[b].cpu().numpy())
                    ).to(output.device)

            probs = torch.softmax(output, dim=1)
            b_loss = boundary_loss(probs, dist_maps)
            return loss + 0.1 * b_loss

        weights = [1 / (2 ** i) for i in range(self.num_segmentation_outputs)]
        weights = [w / sum(weights) for w in weights]
        return MultipleOutputLoss2(combined_loss, weights)


if __name__ == "__main__":
    # 用法：
    # nnUNetv2_train <task> 3d_fullres all -tr nnUNetTrainerBoundary
    pass
