import warnings

from .handlers import handlers
from .utils.trace import trace
import pandas as pd
import numpy as np
__all__ = ['profiler']

def profiler(model,system,unit,densities,intermediate_on_chip=False, args=(), kwargs=None):
    results = dict()
    op_result_list = []
    graph = trace(model, args, kwargs)
    flag = 0
    for i, node in enumerate(graph.nodes):
        for operators, func in handlers:
            if isinstance(operators,str):
                operators = [operators]
            