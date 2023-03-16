import torch
from .quant_layer import QuantModule
from .quant_block import BaseQuantBlock
from .quant_model import QuantModel
from typing import Union


def set_act_quantize_params(
    module: Union[QuantModel, QuantModule, BaseQuantBlock],
    cali_data,
    awq: bool = False,
    order: str = "before",
    batch_size: int = 256,
):
    weight_quant, act_quant = act_get_quant_state(order, awq)
    module.set_quant_state(weight_quant, act_quant)

    for t in module.modules():
        if isinstance(t, (QuantModule, BaseQuantBlock)):
            t.act_quantizer.set_inited(False)

    """set or init step size and zero point in the activation quantizer"""
    if not isinstance(cali_data, (tuple, list)):
        batch_size = min(batch_size, cali_data.size(0))
        with torch.no_grad():
            for i in range(int(cali_data.size(0) / batch_size)):
                module(cali_data[i * batch_size : (i + 1) * batch_size].cuda())
        torch.cuda.empty_cache()

        for t in module.modules():
            if isinstance(t, (QuantModule, BaseQuantBlock)):
                t.act_quantizer.set_inited(True)
    else:
        batch_size = min(batch_size, cali_data[0].size(0))
        with torch.no_grad():
            for i in range(int(cali_data[0].size(0) / batch_size)):
                module(
                    *[
                        _[i * batch_size : (i + 1) * batch_size].cuda()
                        for _ in cali_data
                    ]
                )
        torch.cuda.empty_cache()

        for t in module.modules():
            if isinstance(t, (QuantModule, BaseQuantBlock)):
                t.act_quantizer.set_inited(True)


def act_get_quant_state(order, awq):
    if order == "before":
        weight_quant, act_quant = False, True
    elif order == "after":
        weight_quant, act_quant = awq, True
    elif order == "together":
        weight_quant, act_quant = True, True
    else:
        raise NotImplementedError
    return weight_quant, act_quant
