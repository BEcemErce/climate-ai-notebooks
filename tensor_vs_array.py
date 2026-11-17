import numpy as np
import torch

# 1. NumPy to PyTorch (Zero memory copy)
np_array = np.array([1.0, 2.0, 3.0])
tensor = torch.from_numpy(np_array)

# 2. PyTorch to NumPy
tensor_cpu = torch.tensor([4.0, 5.0])
back_to_np = tensor_cpu.numpy()

# Note: If a tensor requires gradients or is on a GPU, 
# you must detach it and move it to the CPU first:
gpu_tensor = torch.tensor([1.0], requires_grad=True).cuda()
safe_np = gpu_tensor.detach().cpu().numpy()