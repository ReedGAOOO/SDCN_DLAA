import torch

# 创建一个张量
x = torch.randn(2, 3)
print(f"Tensor x: {x}")

# 检查 CUDA 是否可用
if torch.cuda.is_available():
    print("CUDA is available!")
    # 将张量移动到 GPU
    x = x.cuda()
    print(f"Tensor x on GPU: {x}")
else:
    print("CUDA is not available.")

print("PyTorch seems to be working correctly.")