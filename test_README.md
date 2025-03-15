# SMAN Layers 兼容性测试

本测试套件旨在验证基于 PaddlePaddle 的 SMAN_layers.py 与基于 PyTorch Geometric 的 SMAN_layers_pyg.py 两套实现在各模块功能、性能和稳定性方面的完全对应性和有效性。

## 测试内容

测试套件包含三个主要部分：

1. **单元测试**：验证每个函数或层的基本功能与输出形状/数值合理性
   - graph_pooling
   - spatial_embedding
   - aggregate_edges_from_nodes
   - concat_node_edge_feat
   - GATLayer
   - CustomGATConv
   - SGATLayer

2. **集成测试**：验证各模块协同工作时整体功能正确
   - SpatialConv 模块集成测试

3. **压力测试**：在大规模数据和高并发场景中验证系统稳定性、性能和资源占用情况
   - 大规模图数据性能测试
   - 内存使用测试

## 环境要求

运行测试前，请确保已安装以下依赖：

```bash
pip install torch torch_geometric paddle matplotlib memory_profiler
```

## 运行测试

### 1. 运行完整测试套件

执行以下命令运行所有测试并生成报告：

```bash
python run_tests.py
```

这将依次执行单元测试、集成测试和压力测试，并生成测试报告。

### 2. 单独运行比较脚本

如果只想比较两种实现的输出结果，可以执行：

```bash
python compare_implementations.py
```

### 3. 单独运行兼容性测试

如果只想运行兼容性测试，可以执行：

```bash
python -m unittest test_sman_layers_compatibility.py
```

## 测试结果

测试执行后，将生成以下结果文件和目录：

1. **test_logs/**：包含测试执行的详细日志
   - test_results.log：测试执行日志
   - performance_large_graph.png：大规模图性能测试结果图表
   - memory_usage.png：内存使用测试结果图表

2. **test_reports/**：包含测试报告
   - test_report.md：测试报告（包含测试摘要、详细结果和结论）

3. **comparison_logs/**：包含两种实现比较的结果
   - performance_comparison.png：性能比较图表

## 测试报告解读

测试报告包含以下几个部分：

1. **摘要**：总结单元测试、集成测试和压力测试的通过/失败情况
2. **详细结果**：列出每个测试的详细结果，包括失败和错误的详细信息
3. **性能测试结果**：展示大规模图数据下的性能测试结果
4. **内存使用结果**：展示不同规模图数据下的内存使用情况
5. **结论**：总结测试结果，判断两种实现是否兼容

## 注意事项

1. 由于 PaddlePaddle 和 PyTorch 在底层实现上存在差异，某些测试可能会显示数值上的微小差异，这是正常的，只要差异在可接受范围内即可。

2. 压力测试可能需要较长时间执行，特别是在处理大规模图数据时。

3. 如果遇到内存不足的问题，可以在 test_sman_layers_compatibility.py 中调整压力测试的图规模。

## 故障排除

如果测试失败，请检查以下几点：

1. 确保已正确安装所有依赖
2. 检查 PaddlePaddle 和 PyTorch 版本是否兼容
3. 查看测试日志以获取详细的错误信息
4. 对于内存错误，尝试减小测试图的规模

## 扩展测试

如需添加新的测试用例，可以：

1. 在 test_sman_layers_compatibility.py 中添加新的测试方法
2. 在 run_tests.py 中将新测试添加到相应的测试套件中
3. 在 compare_implementations.py 中添加新的比较函数

## 贡献

欢迎提交问题报告和改进建议。