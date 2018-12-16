import torch
from focal_loss import BCEFocalLoss

input = torch.rand(3, 44, 44)
target = (torch.rand(3, 44, 44) > 0.5).float()

criterion = BCEFocalLoss()

res = criterion(input, target)
print(res)