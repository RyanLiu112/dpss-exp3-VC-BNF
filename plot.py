import torch
import numpy as np
import matplotlib.pyplot as plt

model_type = 'one'
# model_type = 'one_res'
# model_type = 'many'

train = np.load(f'./{model_type}_train_loss.npy', allow_pickle=True)
valid = np.load(f'./{model_type}_valid_loss.npy', allow_pickle=True)

plt.figure(figsize=(6, 4))
plt.plot(train)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.tight_layout()
plt.savefig(f'./{model_type}_train_loss.png')

print(len(train))
assert len(train) % 60 == 0
print(torch.from_numpy(train[-60:]).cpu().mean().numpy())
print(len(valid))
print(valid[-1].cpu().numpy())
