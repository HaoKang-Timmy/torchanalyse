
# torchanalyse
A pytorch model profiler with information about flops, energy, and e.t.c
# How to use
Please see the files at `/examples ` like `test_linear.py` and `test_transformer.py` for more information.

Basically, we use the `profiler` function in torch analyze.

# How to install

simply

```
pip3 install torchanalyse
```

# What will the result be like

## Result of linear layer

|   | Op Type      | Dimension                      | Bound | C/M ratio            | Op Intensity       | Latency (msec)         | Cycles             | C Effcy | Flops (MFLOP) | Input_a (MB) | Input_w (MB) | Output (MB) | Total Data (MB) | Throughput (Tflops) | Roofline Throughput offchip (Tflops) | Roofline Throughput onchip (Tflops) | Compute Cycles      | Memory Cycles      | Sparsity | Total energy (mJ)  |
|---|--------------|--------------------------------|-------|----------------------|--------------------|------------------------|--------------------|---------|---------------|--------------|--------------|-------------|-----------------|---------------------|--------------------------------------|-------------------------------------|---------------------|--------------------|----------|--------------------|
| 0 | aten::linear | "([1, 16], [32, 16], [1, 32])" | M     | 0.006689895470383274 | 0.9142857142857143 | 1.2444444444444445e-06 | 1.1697777777777778 | 1.0     | 0.001024      | 1.6e-05      | 0.000512     | 3.2e-05     | 0.00056         | 0.8228571428571428  | 0.8228571428571428                   | 0.8228571428571428                  | 0.00782569105691057 | 1.1697777777777778 | 0.0      | 154980.04707236143 |

For now the `profile` function will provide a datafram with several information for each aten operators. You could see the flops of each at the line of `Flops`.

I may try to refine the datafram structure in the future.
