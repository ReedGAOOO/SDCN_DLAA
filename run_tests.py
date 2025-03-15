import os
import sys
import unittest
import time
import matplotlib.pyplot as plt
from test_sman_layers_compatibility import TestSMANLayersCompatibility

def run_unit_tests():
    """运行单元测试"""
    print("Running unit tests...")
    
    # 创建测试套件
    suite = unittest.TestSuite()
    
    # 添加单元测试
    suite.addTest(TestSMANLayersCompatibility('test_graph_pooling'))
    suite.addTest(TestSMANLayersCompatibility('test_spatial_embedding'))
    suite.addTest(TestSMANLayersCompatibility('test_aggregate_edges_from_nodes'))
    suite.addTest(TestSMANLayersCompatibility('test_concat_node_edge_feat'))
    suite.addTest(TestSMANLayersCompatibility('test_gat_layer'))
    suite.addTest(TestSMANLayersCompatibility('test_custom_gat_conv'))
    suite.addTest(TestSMANLayersCompatibility('test_sgat_layer'))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result

def run_integration_tests():
    """运行集成测试"""
    print("\nRunning integration tests...")
    
    # 创建测试套件
    suite = unittest.TestSuite()
    
    # 添加集成测试
    suite.addTest(TestSMANLayersCompatibility('test_spatial_conv_integration'))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result

def run_stress_tests():
    """运行压力测试"""
    print("\nRunning stress tests...")
    
    # 创建测试套件
    suite = unittest.TestSuite()
    
    # 添加压力测试
    suite.addTest(TestSMANLayersCompatibility('test_performance_large_graph'))
    suite.addTest(TestSMANLayersCompatibility('test_memory_usage'))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result

def generate_report(unit_result, integration_result, stress_result):
    """生成测试报告"""
    print("\nGenerating test report...")
    
    # 创建报告目录
    os.makedirs('test_reports', exist_ok=True)
    
    # 创建报告文件
    with open('test_reports/test_report.md', 'w') as f:
        f.write("# SMAN Layers 兼容性测试报告\n\n")
        f.write("## 摘要\n\n")
        
        # 单元测试摘要
        f.write("### 单元测试\n\n")
        f.write(f"- 总测试数: {unit_result.testsRun}\n")
        f.write(f"- 通过: {unit_result.testsRun - len(unit_result.errors) - len(unit_result.failures)}\n")
        f.write(f"- 失败: {len(unit_result.failures)}\n")
        f.write(f"- 错误: {len(unit_result.errors)}\n\n")
        
        # 集成测试摘要
        f.write("### 集成测试\n\n")
        f.write(f"- 总测试数: {integration_result.testsRun}\n")
        f.write(f"- 通过: {integration_result.testsRun - len(integration_result.errors) - len(integration_result.failures)}\n")
        f.write(f"- 失败: {len(integration_result.failures)}\n")
        f.write(f"- 错误: {len(integration_result.errors)}\n\n")
        
        # 压力测试摘要
        f.write("### 压力测试\n\n")
        f.write(f"- 总测试数: {stress_result.testsRun}\n")
        f.write(f"- 通过: {stress_result.testsRun - len(stress_result.errors) - len(stress_result.failures)}\n")
        f.write(f"- 失败: {len(stress_result.failures)}\n")
        f.write(f"- 错误: {len(stress_result.errors)}\n\n")
        
        # 详细结果
        f.write("## 详细结果\n\n")
        
        # 单元测试详细结果
        f.write("### 单元测试详情\n\n")
        if unit_result.failures:
            f.write("#### 失败\n\n")
            for test, traceback in unit_result.failures:
                f.write(f"- {test}\n")
                f.write("```\n")
                f.write(traceback)
                f.write("```\n\n")
        
        if unit_result.errors:
            f.write("#### 错误\n\n")
            for test, traceback in unit_result.errors:
                f.write(f"- {test}\n")
                f.write("```\n")
                f.write(traceback)
                f.write("```\n\n")
        
        # 集成测试详细结果
        f.write("### 集成测试详情\n\n")
        if integration_result.failures:
            f.write("#### 失败\n\n")
            for test, traceback in integration_result.failures:
                f.write(f"- {test}\n")
                f.write("```\n")
                f.write(traceback)
                f.write("```\n\n")
        
        if integration_result.errors:
            f.write("#### 错误\n\n")
            for test, traceback in integration_result.errors:
                f.write(f"- {test}\n")
                f.write("```\n")
                f.write(traceback)
                f.write("```\n\n")
        
        # 压力测试详细结果
        f.write("### 压力测试详情\n\n")
        if stress_result.failures:
            f.write("#### 失败\n\n")
            for test, traceback in stress_result.failures:
                f.write(f"- {test}\n")
                f.write("```\n")
                f.write(traceback)
                f.write("```\n\n")
        
        if stress_result.errors:
            f.write("#### 错误\n\n")
            for test, traceback in stress_result.errors:
                f.write(f"- {test}\n")
                f.write("```\n")
                f.write(traceback)
                f.write("```\n\n")
        
        # 性能测试结果
        f.write("## 性能测试结果\n\n")
        f.write("![性能测试结果](../test_logs/performance_large_graph.png)\n\n")
        
        # 内存使用结果
        f.write("## 内存使用结果\n\n")
        f.write("![内存使用结果](../test_logs/memory_usage.png)\n\n")
        
        # 结论
        f.write("## 结论\n\n")
        total_tests = unit_result.testsRun + integration_result.testsRun + stress_result.testsRun
        total_passed = (unit_result.testsRun - len(unit_result.errors) - len(unit_result.failures)) + \
                      (integration_result.testsRun - len(integration_result.errors) - len(integration_result.failures)) + \
                      (stress_result.testsRun - len(stress_result.errors) - len(stress_result.failures))
        
        if total_passed == total_tests:
            f.write("所有测试均通过。SMAN layers 的 PyG 实现与 PaddlePaddle 实现兼容。\n")
        else:
            f.write(f"{total_passed} / {total_tests} 测试通过。PyG 和 PaddlePaddle 实现之间存在一些兼容性问题需要解决。\n")
    
    print(f"测试报告已生成：test_reports/test_report.md")

def main():
    """主函数"""
    # 运行测试
    unit_result = run_unit_tests()
    integration_result = run_integration_tests()
    stress_result = run_stress_tests()
    
    # 生成报告
    generate_report(unit_result, integration_result, stress_result)
    
    # 运行比较脚本
    print("\n运行实现比较...")
    os.system("python compare_implementations.py")
    
    # 总结
    print("\n测试执行完成。")
    print("查看 test_logs 目录获取详细日志。")
    print("查看 test_reports 目录获取测试报告。")
    print("查看 comparison_logs 目录获取实现比较结果。")

if __name__ == '__main__':
    main()