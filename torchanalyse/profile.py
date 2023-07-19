import warnings

from .utils.operatorlist.operator_list import operator_list
from .utils.trace import trace
import pandas as pd
import numpy as np
__all__ = ['profiler']

def profiler(model,input,system = None,unit = None,densities = None,intermediate_on_chip=False):
    op_result_list = []
    graph = trace(model, input)
    flag = 0
    for i, node in enumerate(graph.nodes):
        if node.operator in operator_list.keys():
            func = operator_list[node.operator]
            if func is not None:
                operator = func(node)
                
                # op_result = {
                #     "Op_name" : node.operator,
                #     "Macs" : operator.get_num_ops(),
                #     "Tensor size" : operator.get_tensors()
                # }
                op_result = operator.get_roofline(system,unit)
                if flag == 0:
                    column = op_result.keys()
                    flag = 1
                op_result_list.append([op_result[c] for c in column])
            # continue
        else:
            warnings.warn('No handlers found: "{}". Skipped.'.format(
                node.operator))
                

    op_result_df = pd.DataFrame(np.array(op_result_list,dtype=object), columns=column, dtype=object)
    return op_result_df