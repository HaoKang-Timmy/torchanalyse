import torch
from torch.nn.modules.transformer import Transformer
from torch.nn.modules.transformer import Transformer
from torchanalyse import profiler, System, Unit

if __name__ == "__main__":
    embed_size = 512
    num_tokens = 30
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
    model = Transformer(embed_size)
    inputs = (
        torch.randn(num_tokens, 1, embed_size),
        torch.randn(num_tokens, 1, embed_size),
    )

    # macs = profile_macs(model, inputs)
    # print('transformer: {:.4g} G'.format(macs / 1e9))
    op_df = profiler(model, inputs, system, unit)
    print(op_df)
    # for item in op_df:
    #     print(item)
    # for key,value in macs.items():
    #     print("key:   ",key)
    #     print("value:   ",value)
