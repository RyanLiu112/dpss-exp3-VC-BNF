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
plt.legend()
plt.tight_layout()
plt.savefig(f'./{model_type}_train_loss.png')
