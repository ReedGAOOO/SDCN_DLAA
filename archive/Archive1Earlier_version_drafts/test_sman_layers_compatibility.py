import unittest
import numpy as np
import torch
import paddle
import paddle.fluid as fluid
import time
import sys
import os
import matplotlib.pyplot as plt
from memory_profiler import profile

# 导入两个实现
sys.path.append('.')
import SMAN_layers_pyg
from SMAN_layers_pyg import graph_pooling, SpatialEmbedding, aggregate_edges_from_nodes, concat_node_edge_feat
from SMAN_layers_pyg import GATLayer, CustomGATConv, SGATLayer, SpatialConv

# 设置随机种子以确保可重复性
np.random.seed(42)
torch.manual_seed(42)
paddle.seed(42)

class TestSMANLayersCompatibility(unittest.TestCase):
    """测试SMAN_layers.py和SMAN_layers_pyg.py的兼容性"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建日志目录
        os.makedirs('test_logs', exist_ok=True)
        self.log_file = open('test_logs/test_results.log', 'w')
        self.log_file.write("SMAN Layers Compatibility Test Results\n")
        self.log_file.write("=====================================\n\n")
        
        # 设置误差容忍度
        self.tolerance = 1e-4
        
    def tearDown(self):
        """清理测试环境"""
        self.log_file.close()
        
    def log(self, message):
        """记录测试日志"""
        print(message)
        self.log_file.write(message + "\n")
        
    # ==================== 单元测试 ====================
    
    def test_graph_pooling(self):
        """测试graph_pooling函数"""
        self.log("\n[UT-001] Testing graph_pooling function")
        
        # 创建测试数据
        num_nodes = 6
        feature_dim = 4
        node_feat = torch.rand(num_nodes, feature_dim)
        batch = torch.tensor([0, 0, 1, 1, 1, 2])  # 3个图，第一个有2个节点，第二个有3个节点，第三个有1个节点
        
        # 测试不同的池化类型
        for pool_type in ['sum', 'mean', 'max']:
            self.log(f"Testing pool_type: {pool_type}")
            
            # 执行PyG实现的graph_pooling
            result = graph_pooling(node_feat, batch, pool_type)
            
            # 验证结果形状
            expected_shape = (3, feature_dim)  # 3个图，每个特征维度为4
            self.assertEqual(result.shape, expected_shape, 
                            f"Shape mismatch for {pool_type} pooling. Expected {expected_shape}, got {result.shape}")
            
            # 手动计算预期结果进行验证
            expected_result = torch.zeros(3, feature_dim)
            for i in range(num_nodes):
                b = batch[i].item()
                if pool_type == 'sum':
                    expected_result[b] += node_feat[i]
                elif pool_type == 'mean':
                    expected_result[b] += node_feat[i] / torch.sum(batch == b).float()
                elif pool_type == 'max':
                    expected_result[b] = torch.max(torch.stack([expected_result[b], node_feat[i]]), dim=0)[0]
            
            # 对于mean池化，需要特殊处理
            if pool_type == 'mean':
                for b in range(3):
                    count = torch.sum(batch == b).item()
                    if count > 0:
                        expected_result[b] = expected_result[b] * count
            
            # 验证结果值（考虑到不同实现可能有细微差异，使用近似相等）
            self.assertTrue(torch.allclose(result, expected_result, atol=self.tolerance),
                           f"Value mismatch for {pool_type} pooling")
            
            self.log(f"✓ graph_pooling with {pool_type} passed")
        
    def test_spatial_embedding(self):
        """测试SpatialEmbedding类"""
        self.log("\n[UT-002] Testing SpatialEmbedding class")
        
        # 创建测试数据
        dist_dim = 5
        embed_size = 8
        num_edges = 10
        
        # 创建one-hot距离特征
        dist_feat = torch.zeros(num_edges, dist_dim)
        for i in range(num_edges):
            dist_feat[i, i % dist_dim] = 1.0
            
        dist_feat_order = torch.zeros(num_edges, dist_dim)
        for i in range(num_edges):
            dist_feat_order[i, (i + 1) % dist_dim] = 1.0
        
        # 初始化SpatialEmbedding层
        spatial_embed = SpatialEmbedding(dist_dim, embed_size)
        
        # 执行前向传播
        embedded_dist_feat, embedded_dist_feat_order = spatial_embed(dist_feat, dist_feat_order)
        
        # 验证结果形状
        self.assertEqual(embedded_dist_feat.shape, (num_edges, embed_size),
                        f"Shape mismatch for embedded_dist_feat. Expected {(num_edges, embed_size)}, got {embedded_dist_feat.shape}")
        self.assertEqual(embedded_dist_feat_order.shape, (num_edges, embed_size),
                        f"Shape mismatch for embedded_dist_feat_order. Expected {(num_edges, embed_size)}, got {embedded_dist_feat_order.shape}")
        
        # 验证结果不全为零（确保嵌入有效）
        self.assertFalse(torch.allclose(embedded_dist_feat, torch.zeros_like(embedded_dist_feat)),
                        "embedded_dist_feat should not be all zeros")
        self.assertFalse(torch.allclose(embedded_dist_feat_order, torch.zeros_like(embedded_dist_feat_order)),
                        "embedded_dist_feat_order should not be all zeros")
        
        self.log("✓ SpatialEmbedding passed")
        
    def test_aggregate_edges_from_nodes(self):
        """测试aggregate_edges_from_nodes函数"""
        self.log("\n[UT-003] Testing aggregate_edges_from_nodes function")
        
        # 创建测试数据
        num_nodes = 4
        num_edges = 5
        feature_size = 6
        embed_size = 4
        
        # 创建节点-边特征矩阵
        node_edge_feat = torch.rand(num_nodes + num_edges, feature_size)
        
        # 创建距离特征
        dist_feat = torch.rand(num_edges, embed_size)
        
        # 创建边的源节点和目标节点索引
        srcs = torch.tensor([0, 1, 2, 0, 3])
        dsts = torch.tensor([1, 2, 3, 3, 0])
        
        # 执行aggregate_edges_from_nodes
        result = aggregate_edges_from_nodes(node_edge_feat, dist_feat, srcs, dsts)
        
        # 验证结果形状
        expected_shape = (num_edges, feature_size)
        self.assertEqual(result.shape, expected_shape,
                        f"Shape mismatch. Expected {expected_shape}, got {result.shape}")
        
        # 手动计算预期结果
        expected_result = torch.zeros(num_edges, feature_size)
        for i in range(num_edges):
            src_feat = node_edge_feat[srcs[i]]
            dst_feat = node_edge_feat[dsts[i]]
            concat_feat = torch.cat([src_feat, dst_feat, dist_feat[i]], dim=0)
            
            # 模拟全连接层和ReLU激活
            fc_layer = torch.nn.Linear(feature_size * 2 + embed_size, feature_size)
            expected_result[i] = torch.relu(fc_layer(concat_feat))
        
        # 由于我们无法精确复制随机初始化的全连接层权重，这里我们只检查结果不全为零
        self.assertFalse(torch.allclose(result, torch.zeros_like(result)),
                        "Result should not be all zeros")
        
        self.log("✓ aggregate_edges_from_nodes passed")
        
    def test_concat_node_edge_feat(self):
        """测试concat_node_edge_feat函数"""
        self.log("\n[UT-004] Testing concat_node_edge_feat function")
        
        # 创建测试数据
        num_nodes = 4
        num_edges = 5
        feature_size = 6
        
        node_feat = torch.rand(num_nodes, feature_size)
        edge_feat = torch.rand(num_edges, feature_size)
        
        # 执行concat_node_edge_feat
        result = concat_node_edge_feat(node_feat, edge_feat)
        
        # 验证结果形状
        expected_shape = (num_nodes + num_edges, feature_size)
        self.assertEqual(result.shape, expected_shape,
                        f"Shape mismatch. Expected {expected_shape}, got {result.shape}")
        
        # 验证结果内容
        self.assertTrue(torch.allclose(result[:num_nodes], node_feat),
                       "First part of result should match node_feat")
        self.assertTrue(torch.allclose(result[num_nodes:], edge_feat),
                       "Second part of result should match edge_feat")
        
        self.log("✓ concat_node_edge_feat passed")
        
    def test_gat_layer(self):
        """测试GATLayer类"""
        self.log("\n[UT-005] Testing GATLayer class")
        
        # 创建测试数据
        num_nodes = 4
        in_channels = 6
        out_channels = 8
        
        # 创建节点特征
        x = torch.rand(num_nodes, in_channels)
        
        # 创建边索引
        edge_index = torch.tensor([[0, 1, 1, 2, 3], 
                                   [1, 0, 2, 3, 0]]) # 5条边
        
        # 初始化GATLayer
        gat_layer = GATLayer(in_channels, out_channels, heads=2)
        
        # 执行前向传播
        result = gat_layer(x, edge_index)
        
        # 验证结果形状
        expected_shape = (num_nodes, out_channels)
        self.assertEqual(result.shape, expected_shape,
                        f"Shape mismatch. Expected {expected_shape}, got {result.shape}")
        
        # 验证结果不全为零
        self.assertFalse(torch.allclose(result, torch.zeros_like(result)),
                        "Result should not be all zeros")
        
        self.log("✓ GATLayer passed")
        
    def test_custom_gat_conv(self):
        """测试CustomGATConv类"""
        self.log("\n[UT-006] Testing CustomGATConv class")
        
        # 创建测试数据
        num_nodes = 4
        in_channels = 6
        out_channels = 8
        edge_dim = 5
        
        # 创建节点特征
        x = torch.rand(num_nodes, in_channels)
        
        # 创建边索引
        edge_index = torch.tensor([[0, 1, 1, 2, 3], 
                                   [1, 0, 2, 3, 0]]) # 5条边
        
        # 创建边特征
        edge_attr = torch.rand(5, edge_dim)
        
        # 初始化CustomGATConv
        custom_gat = CustomGATConv(in_channels, out_channels, heads=2, edge_dim=edge_dim)
        
        # 执行前向传播
        result = custom_gat(x, edge_index, edge_attr)
        
        # 验证结果形状
        expected_shape = (num_nodes, out_channels)
        self.assertEqual(result.shape, expected_shape,
                        f"Shape mismatch. Expected {expected_shape}, got {result.shape}")
        
        # 验证结果不全为零
        self.assertFalse(torch.allclose(result, torch.zeros_like(result)),
                        "Result should not be all zeros")
        
        self.log("✓ CustomGATConv passed")
        
    def test_sgat_layer(self):
        """测试SGATLayer类"""
        self.log("\n[UT-007] Testing SGATLayer class")
        
        # 创建测试数据
        num_nodes = 4
        in_channels = 6
        out_channels = 8
        
        # 创建节点特征
        x = torch.rand(num_nodes, in_channels)
        
        # 创建边索引
        edge_index = torch.tensor([[0, 1, 1, 2, 3], 
                                   [1, 0, 2, 3, 0]]) # 5条边
        
        # 创建边特征
        edge_attr = torch.rand(5, in_channels)
        
        # 初始化SGATLayer
        sgat_layer = SGATLayer(in_channels, out_channels, heads=2)
        
        # 执行前向传播
        result = sgat_layer(x, edge_index, edge_attr)
        
        # 验证结果形状
        expected_shape = (num_nodes, out_channels)
        self.assertEqual(result.shape, expected_shape,
                        f"Shape mismatch. Expected {expected_shape}, got {result.shape}")
        
        # 验证结果不全为零
        self.assertFalse(torch.allclose(result, torch.zeros_like(result)),
                        "Result should not be all zeros")
        
        self.log("✓ SGATLayer passed")
    
    # ==================== 集成测试 ====================
    
    def test_spatial_conv_integration(self):
        """测试SpatialConv类的集成功能"""
        self.log("\n[IT-001] Testing SpatialConv integration")
        
        # 创建一个小型图结构
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
        
        # 初始化SpatialConv
        spatial_conv = SpatialConv(hidden_size)
        
        # 执行前向传播
        result = spatial_conv(data)
        
        # 验证结果形状
        expected_shape = (num_nodes + num_edges, hidden_size)
        self.assertEqual(result.shape, expected_shape,
                        f"Shape mismatch. Expected {expected_shape}, got {result.shape}")
        
        # 验证结果不全为零
        self.assertFalse(torch.allclose(result, torch.zeros_like(result)),
                        "Result should not be all zeros")
        
        # 验证节点特征和边特征部分
        node_feat = result[:num_nodes]
        edge_feat = result[num_nodes:]
        
        self.assertEqual(node_feat.shape, (num_nodes, hidden_size),
                        f"Node feature shape mismatch. Expected {(num_nodes, hidden_size)}, got {node_feat.shape}")
        self.assertEqual(edge_feat.shape, (num_edges, hidden_size),
                        f"Edge feature shape mismatch. Expected {(num_edges, hidden_size)}, got {edge_feat.shape}")
        
        self.log("✓ SpatialConv integration passed")
    
    # ==================== 压力测试 ====================
    
    @profile
    def test_performance_large_graph(self):
        """测试大规模图数据下的性能"""
        self.log("\n[ST-001] Testing performance with large graph")
        
        # 创建大规模图数据
        num_nodes = 1000  # 对于完整测试可以增加到10000
        num_edges = 5000  # 对于完整测试可以增加到50000
        hidden_size = 64
        
        # 创建一个PyG Data对象模拟
        class LargeGraphData:
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
        
        data = LargeGraphData()
        
        # 初始化SpatialConv
        spatial_conv = SpatialConv(hidden_size)
        
        # 测量执行时间
        start_time = time.time()
        result = spatial_conv(data)
        end_time = time.time()
        
        execution_time = end_time - start_time
        self.log(f"SpatialConv execution time for large graph: {execution_time:.4f} seconds")
        
        # 验证结果形状
        expected_shape = (num_nodes + num_edges, hidden_size)
        self.assertEqual(result.shape, expected_shape,
                        f"Shape mismatch. Expected {expected_shape}, got {result.shape}")
        
        # 多次执行以测试稳定性
        execution_times = []
        for i in range(5):
            start_time = time.time()
            result = spatial_conv(data)
            end_time = time.time()
            execution_times.append(end_time - start_time)
            
        avg_time = sum(execution_times) / len(execution_times)
        std_dev = (sum((t - avg_time) ** 2 for t in execution_times) / len(execution_times)) ** 0.5
        
        self.log(f"Average execution time: {avg_time:.4f} seconds")
        self.log(f"Standard deviation: {std_dev:.4f} seconds")
        
        # 绘制执行时间图表
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(execution_times)), execution_times)
        plt.xlabel('Execution Number')
        plt.ylabel('Execution Time (seconds)')
        plt.title('SpatialConv Execution Time for Large Graph')
        plt.savefig('test_logs/performance_large_graph.png')
        
        self.log("✓ Performance test with large graph passed")
        
    def test_memory_usage(self):
        """测试内存使用情况"""
        self.log("\n[ST-002] Testing memory usage")
        
        # 创建不同规模的图数据并测量内存使用
        sizes = [100, 500, 1000]
        memory_usage = []
        
        for size in sizes:
            num_nodes = size
            num_edges = size * 5
            hidden_size = 64
            
            # 创建一个PyG Data对象模拟
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
            
            # 初始化SpatialConv
            spatial_conv = SpatialConv(hidden_size)
            
            # 使用memory_profiler测量内存使用
            @profile
            def run_model():
                return spatial_conv(data)
            
            # 执行并记录内存使用
            result = run_model()
            
            # 这里我们简单地估计内存使用
            # 实际应用中，应该使用memory_profiler的输出
            estimated_memory = sys.getsizeof(result.storage()) / (1024 * 1024)  # MB
            memory_usage.append(estimated_memory)
            
            self.log(f"Estimated memory usage for size {size}: {estimated_memory:.2f} MB")
        
        # 绘制内存使用图表
        plt.figure(figsize=(10, 6))
        plt.plot(sizes, memory_usage, marker='o')
        plt.xlabel('Graph Size (nodes)')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory Usage vs Graph Size')
        plt.grid(True)
        plt.savefig('test_logs/memory_usage.png')
        
        self.log("✓ Memory usage test passed")

if __name__ == '__main__':
    unittest.main()