import torch.nn as nn
import torch.nn.functional as F
  

class NormalizedMSE(nn.Module):
  def __init__(self, norm_type = "mean-std", *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.norm_type = norm_type

  def mi_ma_norm(self, x):
    mi, ma = x.min(dim=-1, keepdims= True)[0], x.max(dim=-1, keepdims=True)[0]
    return (x-mi)/(ma - mi)
  
  def mean_std_norm(self, x):
    E, s = x.mean(dim=-1, keepdims=True),  x.std(dim=-1, keepdims=True)
    return (x-E)/s

  def forward(self, input, target):
    batch = target.shape[0]
    flat_input = input.reshape((batch, -1))
    flat_target = target.reshape((batch, -1))
    if self.norm_type == "mean-std":
      return F.mse_loss(self.mean_std_norm(flat_input), self.mean_std_norm(flat_target))
    return F.mse_loss(self.mi_ma_norm(flat_input), self.mi_ma_norm(flat_target))
