from torchanalyse import profiler, System, Unit
import torch.nn as nn
import torch
import pandas as pd
import torchvision
if __name__ == "__main__":
    unit = Unit()
    system = System(
        unit,
        frequency=940,
        flops=123,
        onchip_mem_bw=900,
        pe_min_density_support=0.0001,
        accelerator_type="structured",
        model_on_chip_mem_implications=False,
        on_chip_mem_size=32,
    )
    in_features = 16
    out_features = 32

    model = torchvision.models.AlexNet()
    inputs = torch.randn(1, 3,224,224)
    df = profiler(model, inputs, system, unit)
    df.to_csv("./test.csv")

    print(df)
