import torch

t1 = torch.tensor([[0.1,0.2],[0.2,0.3],[0.3,0.1]])


print(torch.sort(t1, dim=0, descending=True))