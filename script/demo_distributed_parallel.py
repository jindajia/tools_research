import torch
import torch.distributed as dist
from torch import nn

def custom_compress(tensor):
    # 模拟压缩函数
    return tensor / 4.0

def custom_decompress(tensor):
    # 模拟解压缩函数
    return tensor * 4.0

def train(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # 模型和优化器初始化
    model = nn.Linear(10, 10).to(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(10000):  # 假设我们有10000个epochs
        # 假设的数据和标签
        inputs = torch.randn(20, 10).to(rank)
        labels = torch.randn(20, 10).to(rank)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss_fn = nn.MSELoss()
        loss = loss_fn(outputs, labels)
        loss.backward()

        # 在进行梯度平均之前应用压缩
        for param in model.parameters():
            param.grad = custom_compress(param.grad)

        # 执行all-reduce操作进行梯度平均
        for param in model.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= world_size

        # 在更新参数之前应用解压缩
        for param in model.parameters():
            param.grad = custom_decompress(param.grad)

        # 参数更新
        optimizer.step()

        # 每隔1000个iteration保存参数和梯度
        if epoch % 1000 == 0:
            if rank == 0:  # 只在一个进程中保存
                torch.save({f'param_{name}': param for name, param in enumerate(model.parameters())}, f"params_epoch_{epoch}.pth")
                torch.save({f'grad_{name}': param.grad for name, param in enumerate(model.parameters())}, f"grads_epoch_{epoch}.pth")

# 启动训练
world_size = 2
for rank in range(world_size):
    train(rank, world_size)
