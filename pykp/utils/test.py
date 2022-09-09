import torch
from masked_loss import masked_cross_entropy
class_dist = torch.tensor([[[0.1, 0.2, 0.3], [0.6, 0.5, 0.4]]])
target = torch.tensor([[2, 1]])
trg_mask = None
loss_scales = torch.ones(2)*5
scale_indices = torch.tensor([2, 0])
masked_cross_entropy(class_dist, target, trg_mask, scale_indices=scale_indices)