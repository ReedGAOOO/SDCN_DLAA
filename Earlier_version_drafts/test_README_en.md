# SMAN Layers Compatibility Test

This test suite aims to verify the complete correspondence and effectiveness of the two implementations, `SMAN_layers.py` based on PaddlePaddle and `SMAN_layers_pyg.py` based on PyTorch Geometric, in terms of module functionality, performance, and stability.

## Test Content

The test suite includes three main parts:

1.  **Unit Tests**: Verify the basic functionality and the reasonableness of output shapes/values for each function or layer.
    *   graph_pooling
    *   spatial_embedding
    *   aggregate_edges_from_nodes
    *   concat_node_edge_feat
    *   GATLayer
    *   CustomGATConv
    *   SGATLayer

2.  **Integration Tests**: Verify the overall functional correctness when modules work together.
    *   SpatialConv module integration test

3.  **Stress Tests**: Verify system stability, performance, and resource usage under large-scale data and high concurrency scenarios.
    *   Large-scale graph data performance test
    *   Memory usage test

## Environment Requirements

Before running the tests, please ensure the following dependencies are installed:

```bash
pip install torch torch_geometric paddle matplotlib memory_profiler
```

## Running Tests

### 1. Run the Complete Test Suite

Execute the following command to run all tests and generate reports:

```bash
python run_tests.py
```

This will sequentially execute unit tests, integration tests, and stress tests, and generate test reports.

### 2. Run Comparison Script Separately

If you only want to compare the output results of the two implementations, you can execute:

```bash
python compare_implementations.py
```

### 3. Run Compatibility Tests Separately

If you only want to run the compatibility tests, you can execute:

```bash
python -m unittest test_sman_layers_compatibility.py
```

## Test Results

After the tests are executed, the following result files and directories will be generated:

1.  **test_logs/**: Contains detailed logs of test execution.
    *   test_results.log: Test execution log
    *   performance_large_graph.png: Performance test result chart for large-scale graphs
    *   memory_usage.png: Memory usage test result chart

2.  **test_reports/**: Contains test reports.
    *   test_report.md: Test report (including test summary, detailed results, and conclusion)

3.  **comparison_logs/**: Contains the comparison results of the two implementations.
    *   performance_comparison.png: Performance comparison chart

## Interpreting Test Reports

The test report includes the following sections:

1.  **Summary**: Summarizes the pass/fail status of unit tests, integration tests, and stress tests.
2.  **Detailed Results**: Lists the detailed results for each test, including details of failures and errors.
3.  **Performance Test Results**: Shows the performance test results under large-scale graph data.
4.  **Memory Usage Results**: Shows the memory usage under different scales of graph data.
5.  **Conclusion**: Summarizes the test results and determines if the two implementations are compatible.

## Notes

1.  Due to differences in the underlying implementations of PaddlePaddle and PyTorch, some tests may show minor numerical discrepancies. This is normal as long as the differences are within an acceptable range.

2.  Stress tests may take a long time to execute, especially when processing large-scale graph data.

3.  If you encounter out-of-memory issues, you can adjust the graph scale for stress tests in `test_sman_layers_compatibility.py`.

## Troubleshooting

If tests fail, please check the following points:

1.  Ensure all dependencies are correctly installed.
2.  Check if the PaddlePaddle and PyTorch versions are compatible.
3.  Review the test logs for detailed error messages.
4.  For memory errors, try reducing the scale of the test graph.

## Extending Tests

To add new test cases, you can:

1.  Add new test methods in `test_sman_layers_compatibility.py`.
2.  Add the new tests to the corresponding test suite in `run_tests.py`.
3.  Add new comparison functions in `compare_implementations.py`.

## Contribution

Contributions, issue reports, and improvement suggestions are welcome.