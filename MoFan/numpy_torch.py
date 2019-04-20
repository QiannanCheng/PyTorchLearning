import torch
import numpy as np

# torch_tensor <--> numpy_array
np_data = np.arange(6).reshape((2, 3))
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()

print(
    '\nnumpy', np_data,
    '\ntorch', torch_data,
    '\ntensor2array', tensor2array,
)

# basic operation
data = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data)  # 32bit

print(
    '\nabs',
    '\nnumpy: ', np.abs(data),        # [1 2 1 2]
    '\ntorch: ', torch.abs(tensor),   # [1 2 1 2]
    '\nnumpy: ', np.sin(data),
    '\ntorch: ', torch.sin(tensor),
    '\nnumpy: ', np.mean(data),
    '\ntorch: ', torch.mean(tensor),
)

# matrix operations
data = [[1,2],[3,4]]
tensor = torch.FloatTensor(data)  # 32-bit floating point

print(
    '\nnumpy:', np.matmul(data, data),
    '\ntorch:', torch.mm(tensor, tensor)
)

data = np.array(data)
print(
    '\nnumpy:', data.dot(data),
    '\ntorch:',
)

