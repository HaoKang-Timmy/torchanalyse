
# torchanalyse
A pytorch model profiler with information about flops, energy, and e.t.c
# How to use
Please see `test_linear.py` and `test_transformer.py` for more information.

Basically, we use the `profiler` function in torch analyse.

# What will the result be like

## Result of linear layer

|   | Op Type      | Dimension                      | Bound | C/M ratio            | Op Intensity       | Latency (msec)         | Cycles             | C Effcy | Flops (MFLOP) | Input_a (MB) | Input_w (MB) | Output (MB) | Total Data (MB) | Throughput (Tflops) | Roofline Throughput offchip (Tflops) | Roofline Throughput onchip (Tflops) | Compute Cycles      | Memory Cycles      | Sparsity | Total energy (mJ)  |
|---|--------------|--------------------------------|-------|----------------------|--------------------|------------------------|--------------------|---------|---------------|--------------|--------------|-------------|-----------------|---------------------|--------------------------------------|-------------------------------------|---------------------|--------------------|----------|--------------------|
| 0 | aten::linear | "([1, 16], [32, 16], [1, 32])" | M     | 0.006689895470383274 | 0.9142857142857143 | 1.2444444444444445e-06 | 1.1697777777777778 | 1.0     | 0.001024      | 1.6e-05      | 0.000512     | 3.2e-05     | 0.00056         | 0.8228571428571428  | 0.8228571428571428                   | 0.8228571428571428                  | 0.00782569105691057 | 1.1697777777777778 | 0.0      | 154980.04707236143 |

## Result of Transformer

![image-20230720032249955](/Users/catbeta/Library/Application Support/typora-user-images/image-20230720032249955.png)
