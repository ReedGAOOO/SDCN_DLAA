"""
PaddlePaddle 和 PyTorch 数据格式适配器

此模块提供了在 PaddlePaddle 和 PyTorch 之间转换数据格式的工具函数，
用于在测试中比较两种实现的输出结果。
"""

import numpy as np
import torch
import paddle
import paddle.fluid as fluid

def paddle_to_torch(paddle_tensor):
    """
    将 PaddlePaddle 张量转换为 PyTorch 张量
    
    Args:
        paddle_tensor: PaddlePaddle 张量
        
    Returns:
        torch.Tensor: 转换后的 PyTorch 张量
    """
    if isinstance(paddle_tensor, (paddle.Tensor, fluid.LoDTensor)):
        # 转换为 numpy 数组
        numpy_array = paddle_tensor.numpy()
        # 转换为 PyTorch 张量
        return torch.from_numpy(numpy_array)
    else:
        raise TypeError(f"不支持的类型: {type(paddle_tensor)}")

def torch_to_paddle(torch_tensor):
    """
    将 PyTorch 张量转换为 PaddlePaddle 张量
    
    Args:
        torch_tensor: PyTorch 张量
        
    Returns:
        paddle.Tensor: 转换后的 PaddlePaddle 张量
    """
    if isinstance(torch_tensor, torch.Tensor):
        # 转换为 numpy 数组
        numpy_array = torch_tensor.detach().cpu().numpy()
        # 转换为 PaddlePaddle 张量
        return paddle.to_tensor(numpy_array)
    else:
        raise TypeError(f"不支持的类型: {type(torch_tensor)}")

def create_paddle_lod_tensor(data, lod_info):
    """
    创建 PaddlePaddle LoD 张量
    
    Args:
        data: 数据，可以是 numpy 数组或 PyTorch 张量
        lod_info: LoD 信息，例如 [[0, 2, 5, 6]] 表示第一个序列长度为 2，第二个为 3，第三个为 1
        
    Returns:
        fluid.LoDTensor: 创建的 LoD 张量
    """
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    
    # 创建 LoD 张量
    lod_tensor = fluid.LoDTensor()
    lod_tensor.set(data, fluid.CPUPlace())
    lod_tensor.set_lod(lod_info)
    
    return lod_tensor

def batch_to_lod(batch_tensor, batch_indices):
    """
    将 PyTorch 的 batch 张量转换为 PaddlePaddle 的 LoD 张量
    
    Args:
        batch_tensor: PyTorch 张量，形状为 [num_nodes, feature_dim]
        batch_indices: 批次索引，形状为 [num_nodes]，指示每个节点属于哪个图
        
    Returns:
        fluid.LoDTensor: 转换后的 LoD 张量
    """
    # 获取批次数量
    num_batches = batch_indices.max().item() + 1
    
    # 计算 LoD 信息
    lod_info = [[0]]
    for i in range(num_batches):
        # 计算当前批次的节点数量
        batch_size = (batch_indices == i).sum().item()
        lod_info[0].append(lod_info[0][-1] + batch_size)
    
    # 重新排序节点，按批次顺序
    sorted_indices = torch.argsort(batch_indices)
    sorted_tensor = batch_tensor[sorted_indices]
    
    # 创建 LoD 张量
    return create_paddle_lod_tensor(sorted_tensor.numpy(), lod_info)

def edge_index_to_paddle_graph(edge_index, num_nodes):
    """
    将 PyTorch Geometric 的边索引转换为 PaddlePaddle 的图结构
    
    Args:
        edge_index: PyTorch 张量，形状为 [2, num_edges]，表示边的源节点和目标节点
        num_nodes: 节点数量
        
    Returns:
        dict: PaddlePaddle 图结构，包含 'edge_src'、'edge_dst' 和 'num_nodes'
    """
    edge_src = edge_index[0].numpy()
    edge_dst = edge_index[1].numpy()
    
    return {
        'edge_src': edge_src,
        'edge_dst': edge_dst,
        'num_nodes': num_nodes
    }

def compare_tensors(torch_tensor, paddle_tensor, tolerance=1e-4):
    """
    比较 PyTorch 张量和 PaddlePaddle 张量是否近似相等
    
    Args:
        torch_tensor: PyTorch 张量
        paddle_tensor: PaddlePaddle 张量
        tolerance: 容忍误差
        
    Returns:
        bool: 如果两个张量近似相等，则返回 True，否则返回 False
    """
    # 转换为 numpy 数组
    torch_array = torch_tensor.detach().cpu().numpy()
    paddle_array = paddle_tensor.numpy()
    
    # 检查形状是否相同
    if torch_array.shape != paddle_array.shape:
        print(f"形状不同: torch={torch_array.shape}, paddle={paddle_array.shape}")
        return False
    
    # 计算相对误差
    abs_diff = np.abs(torch_array - paddle_array)
    abs_paddle = np.abs(paddle_array)
    max_paddle = np.max(abs_paddle)
    
    if max_paddle < tolerance:
        # 如果 paddle 值接近零，使用绝对误差
        max_diff = np.max(abs_diff)
        is_close = max_diff < tolerance
    else:
        # 否则使用相对误差
        rel_diff = abs_diff / (abs_paddle + tolerance)
        max_rel_diff = np.max(rel_diff)
        is_close = max_rel_diff < tolerance
    
    if not is_close:
        print(f"最大相对误差: {max_rel_diff if 'max_rel_diff' in locals() else 'N/A'}")
        print(f"最大绝对误差: {np.max(abs_diff)}")
    
    return is_close

def create_mock_paddle_graph(edge_index, num_nodes):
    """
    创建用于测试的 PaddlePaddle 图包装器
    
    Args:
        edge_index: PyTorch 张量，形状为 [2, num_edges]，表示边的源节点和目标节点
        num_nodes: 节点数量
        
    Returns:
        object: 模拟的 PaddlePaddle 图包装器
    """
    # 这里我们创建一个模拟的图包装器，实际使用时需要替换为真实的 PGL 图包装器
    class MockPaddleGraph:
        def __init__(self, edge_src, edge_dst, num_nodes):
            self.edge_src = edge_src
            self.edge_dst = edge_dst
            self.num_nodes = num_nodes
        
        def send(self, send_func, nfeat_list=None, efeat_list=None):
            # 模拟 send 操作
            return {"message": "模拟的消息"}
        
        def recv(self, msg, reduce_func):
            # 模拟 recv 操作
            return paddle.zeros([self.num_nodes, 10])
    
    edge_src = edge_index[0].numpy()
    edge_dst = edge_index[1].numpy()
    
    return MockPaddleGraph(edge_src, edge_dst, num_nodes)

def prepare_test_data_for_both_frameworks():
    """
    准备用于测试的数据，同时适用于 PyTorch 和 PaddlePaddle
    
    Returns:
        tuple: (torch_data, paddle_data) 包含两种框架的测试数据
    """
    # 创建基本测试数据
    num_nodes = 5
    num_edges = 6
    feature_dim = 8
    
    # PyTorch 数据
    torch_data = {
        'node_feat': torch.rand(num_nodes, feature_dim),
        'edge_index': torch.tensor([[0, 1, 1, 2, 3, 4], 
                                   [1, 0, 2, 3, 4, 0]]),
        'edge_feat': torch.rand(num_edges, feature_dim),
        'batch': torch.tensor([0, 0, 1, 1, 2])
    }
    
    # 转换为 PaddlePaddle 数据
    paddle_data = {
        'node_feat': torch_to_paddle(torch_data['node_feat']),
        'edge_src': torch_data['edge_index'][0].numpy(),
        'edge_dst': torch_data['edge_index'][1].numpy(),
        'edge_feat': torch_to_paddle(torch_data['edge_feat']),
        'graph_lod': [[0, 2, 4, 5]]  # 对应 batch [0, 0, 1, 1, 2]
    }
    
    return torch_data, paddle_data