import numpy as np
import torch
import paddle
import paddle.fluid as fluid
import pgl
import sys
import matplotlib.pyplot as plt
import os

# 导入两个实现
sys.path.append('.')
import SMAN_layers
import SMAN_layers_pyg
from SMAN_layers_pyg import graph_pooling, SpatialEmbedding, aggregate_edges_from_nodes, concat_node_edge_feat
from SMAN_layers_pyg import GATLayer, CustomGATConv, SGATLayer, SpatialConv

# 设置随机种子以确保可重复性
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
paddle.framework.random._manual_program_seed(42)
paddle.framework.random._manual_program_seed(42)
paddle.seed(42)

# 创建日志目录
os.makedirs('comparison_logs', exist_ok=True)

def compare_graph_pooling():
    """比较graph_pooling函数的实现"""
    # 导入适配器函数
    from paddle_torch_adapter import batch_to_lod, paddle_to_torch, compare_tensors
    
    print("\nComparing graph_pooling implementations")
    
    # 创建测试数据
    num_nodes = 6
    feature_dim = 4
    
    # PyTorch数据
    node_feat_torch = torch.rand(num_nodes, feature_dim)
    batch_torch = torch.tensor([0, 0, 1, 1, 1, 2])
    
    # 转换为PaddlePaddle数据
    node_feat_paddle = paddle.to_tensor(node_feat_torch.numpy())
    # 创建LoD信息，对应batch [0,0,1,1,1,2]
    graph_lod = [[0, 2, 5, 6]]
    
    # 测试不同的池化类型
    for pool_type in ['sum', 'mean', 'max']:
        print(f"Testing pool_type: {pool_type}")
        
        # 执行PyG实现的graph_pooling
        result_torch = graph_pooling(node_feat_torch, batch_torch, pool_type)
        
        # 打印PyG结果
        print(f"PyG result shape: {result_torch.shape}")
        print(f"PyG result (first few values): {result_torch[:2]}")
        
        # 尝试执行PaddlePaddle实现
        try:
            # 执行PaddlePaddle实现的graph_pooling
            # 首先创建LoD张量
            node_feat_lod = fluid.create_lod_tensor(node_feat_paddle.numpy(), graph_lod, fluid.CPUPlace())
            
            # 执行graph_pooling
            with fluid.dygraph.guard():
                result_paddle = SMAN_layers.graph_pooling(node_feat_lod, graph_lod, pool_type)
                
                # 转换为PyTorch张量进行比较
                result_paddle_torch = paddle_to_torch(result_paddle)
                
                # 比较结果
                is_close = compare_tensors(result_torch, result_paddle_torch)
                print(f"Results are close: {is_close}")
                
                # 打印PaddlePaddle结果
                print(f"PaddlePaddle result shape: {result_paddle_torch.shape}")
                print(f"PaddlePaddle result (first few values): {result_paddle_torch[:2]}")
        except Exception as e:
            print(f"Error executing PaddlePaddle implementation: {e}")

def compare_spatial_embedding():
    """比较spatial_embedding实现"""
    print("\nComparing spatial_embedding implementations")
    
    # 创建测试数据
    dist_dim = 5
    embed_size = 8
    num_edges = 10
    
    # PyTorch数据
    dist_feat_torch = torch.zeros(num_edges, dist_dim)
    for i in range(num_edges):
        dist_feat_torch[i, i % dist_dim] = 1.0
        
    dist_feat_order_torch = torch.zeros(num_edges, dist_dim)
    for i in range(num_edges):
        dist_feat_order_torch[i, (i + 1) % dist_dim] = 1.0
    
    # 初始化PyG的SpatialEmbedding层
    spatial_embed_torch = SpatialEmbedding(dist_dim, embed_size)
    
    # 执行前向传播
    embedded_dist_feat_torch, embedded_dist_feat_order_torch = spatial_embed_torch(dist_feat_torch, dist_feat_order_torch)
    
    print(f"PyG embedded_dist_feat shape: {embedded_dist_feat_torch.shape}")
    print(f"PyG embedded_dist_feat_order shape: {embedded_dist_feat_order_torch.shape}")
    
    # 在实际测试中，应该比较两个实现的结果
    # 由于权重初始化可能不同，可以比较形状和数值范围

def compare_aggregate_edges_from_nodes():
    """比较aggregate_edges_from_nodes实现"""
    print("\nComparing aggregate_edges_from_nodes implementations")
    
    # 创建测试数据
    num_nodes = 4
    num_edges = 5
    feature_size = 6
    embed_size = 4
    
    # PyTorch数据
    node_edge_feat_torch = torch.rand(num_nodes + num_edges, feature_size)
    dist_feat_torch = torch.rand(num_edges, embed_size)
    srcs_torch = torch.tensor([0, 1, 2, 0, 3])
    dsts_torch = torch.tensor([1, 2, 3, 3, 0])
    
    # 执行PyG实现
    result_torch = aggregate_edges_from_nodes(node_edge_feat_torch, dist_feat_torch, srcs_torch, dsts_torch)
    
    print(f"PyG result shape: {result_torch.shape}")
    
    # 在实际测试中，应该比较两个实现的结果

def compare_concat_node_edge_feat():
    """比较concat_node_edge_feat实现"""
    print("\nComparing concat_node_edge_feat implementations")
    
    # 创建测试数据
    num_nodes = 4
    num_edges = 5
    feature_size = 6
    
    # PyTorch数据
    node_feat_torch = torch.rand(num_nodes, feature_size)
    edge_feat_torch = torch.rand(num_edges, feature_size)
    
    # 执行PyG实现
    result_torch = concat_node_edge_feat(node_feat_torch, edge_feat_torch)
    
    print(f"PyG result shape: {result_torch.shape}")
    
    # 在实际测试中，应该比较两个实现的结果

def compare_gat_layer():
    """比较GATLayer实现"""
    print("\nComparing GATLayer implementations")
    
    # 创建测试数据
    num_nodes = 4
    in_channels = 6
    out_channels = 8
    
    # PyTorch数据
    x_torch = torch.rand(num_nodes, in_channels)
    edge_index_torch = torch.tensor([[0, 1, 1, 2, 3], 
                                    [1, 0, 2, 3, 0]])
    
    # 初始化PyG的GATLayer
    gat_layer_torch = GATLayer(in_channels, out_channels, heads=2)
    
    # 执行前向传播
    result_torch = gat_layer_torch(x_torch, edge_index_torch)
    
    print(f"PyG result shape: {result_torch.shape}")
    
    # 在实际测试中，应该比较两个实现的结果

def compare_spatial_conv():
    """比较SpatialConv实现"""
    print("\nComparing SpatialConv implementations")
    
    # 创建测试数据
    num_nodes = 5
    num_edges = 6
    hidden_size = 8
    
    # 创建一个PyG Data对象模拟
    class MockData:
        def __init__(self):
            self.x = torch.rand(num_nodes, hidden_size)
            self.edge_index = torch.tensor([[0, 1, 1, 2, 3, 4], 
                                           [1, 0, 2, 3, 4, 0]])
            self.edge_attr = torch.rand(num_edges, hidden_size)
            self.dist_feat = torch.rand(num_edges, hidden_size)
            self.dist_feat_order = torch.rand(num_edges, hidden_size)
            # 创建edge-to-edge连接
            self.edge_to_edge_index = torch.tensor([[0, 1, 2, 3, 4], 
                                                   [1, 2, 3, 4, 5]])
    
    data = MockData()
    
    # 初始化PyG的SpatialConv
    spatial_conv_torch = SpatialConv(hidden_size)
    
    # 执行前向传播
    result_torch = spatial_conv_torch(data)
    
    print(f"PyG result shape: {result_torch.shape}")
    
    # 在实际测试中，应该比较两个实现的结果

def performance_test():
    """性能测试"""
    print("\nPerformance testing")
    
    # 创建不同规模的图数据
    sizes = [100, 500, 1000]
    execution_times_torch = []
    
    for size in sizes:
        num_nodes = size
        num_edges = size * 5
        hidden_size = 64
        
        # 创建一个PyG Data对象
        class GraphData:
            def __init__(self):
                self.x = torch.rand(num_nodes, hidden_size)
                # 随机生成边索引
                src_nodes = torch.randint(0, num_nodes, (num_edges,))
                dst_nodes = torch.randint(0, num_nodes, (num_edges,))
                self.edge_index = torch.stack([src_nodes, dst_nodes])
                self.edge_attr = torch.rand(num_edges, hidden_size)
                self.dist_feat = torch.rand(num_edges, hidden_size)
                self.dist_feat_order = torch.rand(num_edges, hidden_size)
                # 创建edge-to-edge连接
                src_edges = torch.randint(0, num_edges, (num_edges,))
                dst_edges = torch.randint(0, num_edges, (num_edges,))
                self.edge_to_edge_index = torch.stack([src_edges, dst_edges])
        
        data = GraphData()
        
        # 初始化PyG的SpatialConv
        spatial_conv_torch = SpatialConv(hidden_size)
        
        # 测量执行时间
        import time
        start_time = time.time()
        result_torch = spatial_conv_torch(data)
        end_time = time.time()
        
        execution_time = end_time - start_time
        execution_times_torch.append(execution_time)
        
        print(f"PyG execution time for size {size}: {execution_time:.4f} seconds")
    
    # 绘制执行时间图表
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, execution_times_torch, marker='o', label='PyG')
    plt.xlabel('Graph Size (nodes)')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time vs Graph Size')
    plt.legend()
    plt.grid(True)
    plt.savefig('comparison_logs/performance_comparison.png')
    
    print(f"Performance comparison plot saved to comparison_logs/performance_comparison.png")

if __name__ == '__main__':
    compare_graph_pooling()
    compare_spatial_embedding()
    compare_aggregate_edges_from_nodes()
    compare_concat_node_edge_feat()
    compare_gat_layer()
    compare_spatial_conv()
    performance_test()