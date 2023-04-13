import torch

t1 = torch.tensor([0.1,0.2])
t2 = torch.tensor([0.3,0.2])

sum_ = []
sum_.append(t1)
sum_.append(t2)

print(torch.cat(sum_))