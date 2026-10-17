import torch

X = torch.arange(start = 0, end=12, step=1, dtype=torch.int16).reshape(3,4,1)
Y = torch.arange(start=-12, end=0,step=1, dtype=torch.int16).reshape(3,1,4)
Z = X + Y
# При сложении получим (3,4,4)

print(f"X+Y={X.numpy()}+{Y.numpy()}={Z.numpy()}")
print(f"Размер результирующего тензора {Z.shape}")