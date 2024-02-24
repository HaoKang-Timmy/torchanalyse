import numpy as np
from .operator_base import Operator
import math


# get_gemms left, upper, contract, outer
class Addmm(Operator):              ## Matrix Execution Unit
    # https://pytorch.org/docs/stable/generated/torch.addmm.html
    # D = A * B + C  A(n.m) B(m,p) C(n,p)
    def __init__(self, node, density=(1.0, 1.0, 1.0)):
        super().__init__(node, density)
        self.alu_type = 'mxu'

    def get_tensors(self):
        n, m = self.node.inputs[1].shape
        m, p = self.node.inputs[2].shape
        return (n, m), (m, p), (n, p)

    def get_num_ops(self):
        n, m = self.node.inputs[1].shape
        m, p = self.node.inputs[2].shape
        return n * m * p

    def get_gemms(self):
        n, m = self.node.inputs[1].shape
        m, p = self.node.inputs[2].shape
        return n, p, m, 1


class Addmv(Operator):          ## Vector Execution Unit 
    # https://pytorch.org/docs/stable/generated/torch.addmv.html#torch.addmv
    # matrix * vector p = 1
    def __init__(self, node, density=(1.0, 1.0, 1.0)):
        super().__init__(node, density)
        self.alu_type = 'vxu'

    def get_tensors(self):
        n, m = self.node.inputs[1].shape
        p = 1
        return (n, m), (m, p), (n, p)

    def get_num_ops(self):
        n, m = self.node.inputs[1].shape
        return n * m

    def get_gemms(self):
        n, m = self.node.inputs[1].shape
        p = 1
        return n, p, m, 1


class Bmm(Operator):            ## Matrix Execution Unit
    # batch matmul : https://pytorch.org/docs/stable/generated/torch.bmm.html#torch.bmm
    # Out (B,n,p) = Input (B,n,m) * Mat2(B,m,p)
    def __init__(self, node, density=(1.0, 1.0, 1.0)):
        super().__init__(node, density)
        self.alu_type = 'mxu'

    def get_tensors(self):
        b, n, m = self.node.inputs[0].shape
        b, m, p = self.node.inputs[1].shape
        return (b, n, m), (b, m, p), (b, n, p)

    def get_num_ops(self):
        # print(self.node.inputs[1], self.node.inputs[2])
        b, n, m = self.node.inputs[0].shape
        b, m, p = self.node.inputs[1].shape
        return b * n * m * p

    def get_gemms(self):
        # TODO might have some problems
        b, n, m = self.node.inputs[0].shape
        b, m, p = self.node.inputs[1].shape
        return n, p, m, b


class Baddbmm(Operator):       ## Matrix Execution Unit
    # batch matmul and add A * B + C = D, (b,n,m) * (b,m,p) + (b,n,p) = (b,n,p)
    def __init__(self, node, density=(1.0, 1.0, 1.0)):
        super().__init__(node, density)
        self.alu_type = 'mxu'

    def get_tensors(self):
        b, n, p = self.node.inputs[0].shape
        b, n1, m = self.node.inputs[1].shape
        b, m1, p1 = self.node.inputs[2].shape
        # C A B D
        return (b, n, p), (b, n1, m), (b, m1, p1), (b, n, p)

    def get_num_ops(self):
        b, n, p = self.node.inputs[0].shape
        b, n1, m = self.node.inputs[1].shape
        return b * n * m * p

    def get_gemms(self):
        # TODO might have some problems
        b, n, m = self.node.inputs[1].shape
        b, m, p = self.node.inputs[2].shape
        return n, p, m, b

class ScaledDotProductAttention(Operator):       ## Matrix Execution Unit
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch-nn-functional-scaled-dot-product-attention
    # L = Q * K'
    # W = Softmax(L)
    # O = L * V

    ## Inputs = Q (N,...,L,E) , K(N,...,S,E), K(N,...,S,Ev)
    ## Ouput (N,..., L, Ev)
    # N:Batch size...:Any number of other batch dimensions (optional)
    # S:Source sequence length
    # L:Target sequence length
    # E:Embedding dimension of the query and key
    # Ev:Embedding dimension of the value

    def __init__(self, node, density=(1.0, 1.0, 1.0)):
        super().__init__(node, density)
        self.alu_type = 'mxu'

    def get_tensors(self):
        # tensor = self.node.inputs[0].shape #Q
        # b, l, e = 
         
        tensor = self.node.inputs[1].shape #K
        b, s, e = np.prod(tensor[:-2]), tensor[-2], tensor[-1] 
        tensor = self.node.inputs[2].shape #V
        b, s, ev = np.prod(tensor[:-2]), tensor[-2], tensor[-1]
       
        # Q K V O
        return (self.node.inputs[0].shape, 
               (b, s, e+ev),
                self.node.outputs[0].shape )

    def get_num_ops(self):
        tensor = self.node.inputs[0].shape #Q
        b, l, e = np.prod(tensor[:-2]), tensor[-2], tensor[-1] 
        tensor = self.node.inputs[1].shape #K
        b, s, e = np.prod(tensor[:-2]), tensor[-2], tensor[-1] 
        tensor = self.node.inputs[2].shape #V
        b, s, ev = np.prod(tensor[:-2]), tensor[-2], tensor[-1]

        QK_ops = b*l*s*e
        KV_ops = b*s*l*ev
        return QK_ops + KV_ops
    

    def get_gemms(self):
        # TODO might have some problems
        b, n, m = self.node.inputs[1].shape
        b, m, p = self.node.inputs[2].shape
        return n, p, m, b


class Matmul(Operator):       ## Matrix Execution Unit
    # https://pytorch.org/docs/stable/generated/torch.matmul.html#torch-matmul
    # matmul A * B = C
    def __init__(self, node, density=(1.0, 1.0, 1.0)):
        if node.inputs[0].ndim == 1 and node.inputs[1].ndim == 1:
            self.type = 0
            # aten::matmul([n], [n])
        elif node.inputs[0].ndim == 1 and node.inputs[1].ndim == 2:
            self.type = 1
            # aten::matmul([n], [n, m])
        elif node.inputs[0].ndim == 2 and node.inputs[1].ndim == 1:
            self.type = 2
            # aten::matmul([n, m], [m])
        elif node.inputs[0].ndim == 2 and node.inputs[1].ndim == 2:
            self.type = 3
            # aten::matmul([n, m], [m, p])
        elif node.inputs[0].ndim == 1:
            self.type = 4
            # aten::matmul([n], [..., n, m])
        elif node.inputs[1].ndim == 1:
            self.type = 5
            # aten::matmul([..., n, m], [m])
        else:
            self.type = 6
            # aten::matmul([..., n, m], [..., m, p])
        super().__init__(node, density)
        self.alu_type = 'mxu'

    def get_tensors(self):
        return (
            self.node.inputs[0].shape,
            self.node.inputs[1].shape,
            self.node.outputs[0].shape,
        )

    def get_num_ops(self):
        if self.type == 0:
            result = self.node.inputs[0].shape[0]
        elif self.type == 1:
            n, m = self.node.inputs[1].shape
            result = n * m
        elif self.type == 2:
            n, m = self.node.inputs[0].shape
            result = n * m
        elif self.type == 3:
            n, m = self.node.inputs[0].shape
            m, p = self.node.inputs[1].shape
            result = n * m * p
        elif self.type == 4:
            *b, n, m = self.node.inputs[1].shape
            result = math.prod(b) * n * m
        elif self.type == 5:
            *b, n, m = self.node.inputs[0].shape
            result = math.prod(b) * n * m
        elif self.type == 6:
            *b, n, p = self.node.outputs[0].shape
            *_, n, m = self.node.inputs[0].shape
            *_, m, p = self.node.inputs[1].shape
            result = math.prod(b) * n * m * p
        return result

    def get_gemms(self):
        # TODO might have some problems
        if self.type == 0:
            contrat = self.node.inputs[0].shape[0]  # n
            return 1, 1, contrat, 1
        elif self.type == 1:
            n, m = self.node.inputs[1].shape
            return 1, m, n, 1
        elif self.type == 2:
            n, m = self.node.inputs[0].shape
            return n, 1, m, 1
        elif self.type == 3:
            n, m = self.node.inputs[0].shape
            m, p = self.node.inputs[1].shape
            return n, p, m, 1
        elif self.type == 4:
            # aten::matmul([n], [..., n, m])
            n = self.node.inputs[0].shape[0]
            *b, n, m = self.node.inputs[1].shape
            return 1, m, n, math.prod(b)
        elif self.type == 5:
            # aten::matmul([..., n, m], [m])
            m = self.node.inputs[0].shape[0]
            *b, n, m = self.node.inputs[1].shape
            return n, 1, m, math.prod(b)
        elif self.type == 6:
            # aten::matmul([..., n, m], [..., m, p])
            *a, n, m = self.node.inputs[0].shape
            *b, m, p = self.node.inputs[1].shape
            return n, p, m, math.prod(b)


class Mul(Operator):          ## Vector Execution Unit
    # A * b, b is constant
    def __init__(self, node, density=(1.0, 1.0, 1.0)):
        super().__init__(node, density)
        # print(node.inputs[0].shape,node.inputs[1].shape,node.outputs[0].shape)
        self.alu_type = 'mxu'

    def get_tensors(self):
        if len(self.node.inputs) == 1: 
            return (
                self.node.inputs[0].shape if self.node.inputs[0].ndim != 0 else 1,
                1,
                self.node.outputs[0].shape if self.node.outputs[0].ndim != 0 else 1,
            )
        else:
            return (
                self.node.inputs[0].shape if self.node.inputs[0].ndim != 0 else 1,
                self.node.inputs[1].shape if self.node.inputs[1].ndim != 0 else 1,
                self.node.outputs[0].shape if self.node.outputs[0].ndim != 0 else 1,
            ) 

    def get_num_ops(self):
        output_shape = self.node.outputs[0].shape
        return math.prod(output_shape)

    def get_gemms(self):
        # TODO might have some problems
        # this seems has nothing to do with gemm
        return 0, 0, 0, 1


class Convolution(Operator):       ## Matrix Execution Unit
    def __init__(self, node, density=(1.0, 1.0, 1.0)):
        super().__init__(node, density)
        self.alu_type = 'mxu'

    def get_tensors(self):
        return (
            self.node.inputs[0].shape,
            self.node.inputs[1].shape,
            self.node.outputs[0].shape,
        )

    def get_num_ops(self):
        if self.node.outputs[0].shape[1] == self.node.inputs[1].shape[0]:
            oc, ic, *ks = self.node.inputs[1].shape
        else:
            ic, oc, *ks = self.node.inputs[1].shape
        os = self.node.outputs[0].shape
        return math.prod(os) * ic * math.prod(ks)

    def get_gemms(self):
        # TODO might have some problems
        # if self.node.outputs[0].shape[1] == self.node.inputs[1].shape[0]:
        #     # weight (H,B,N,D)
        B, H, M, N = self.node.inputs[0].shape
        B, H, M, D = self.node.outputs[0].shape

        return M, D, N, B * H


class Norm(Operator):          ## Vector Execution Unit
    def __init__(self, node, density=(1.0, 1.0, 1.0)):
        super().__init__(node, density)
        self.alu_type = 'vxu'

    def get_tensors(self):
        # input and output
        return self.node.inputs[0].shape, self.node.outputs[0].shape

    def get_num_ops(self):
        # macs
        if self.node.operator in ["aten::batch_norm", "aten::instance_norm"]:
            affine = self.node.inputs[1].shape is not None
        elif self.node.operator in ["aten::layer_norm", "aten::group_norm"]:
            affine = self.node.inputs[2].shape is not None
        else:
            raise ValueError(self.node.operator)
        os = self.node.outputs[0].shape
        return math.prod(os) if affine else 0

    def get_gemms(self):
        # TODO might have some problems
        return 0, 0, 0, 1


class Avg(Operator):          ## Vector Execution Unit
    def __init__(self, node, density=(1.0, 1.0, 1.0)):
        super().__init__(node, density)
        self.alu_type = 'vxu'

    def get_tensors(self):
        # input and output
        return self.node.inputs[0].shape, self.node.outputs[0].shape

    def get_num_ops(self):
        os = self.node.outputs[0].shape
        return math.prod(os)

    def get_gemms(self):
        # TODO might have some problems
        return 0, 0, 0, 1


class Special_Func(Operator):          ## Vector Execution Unit
    def __init__(self, node, density=(1.0, 1.0, 1.0)):
        super().__init__(node, density)
        self.alu_type = 'vxu'

    def get_tensors(self):
        # input and output
        if self.node.inputs[0].shape is not None:
            return self.node.inputs[0].shape, self.node.outputs[0].shape
        else:
            return self.node.inputs[1].shape, self.node.outputs[0].shape

    def get_num_ops(self):
        os = self.node.outputs[0].shape
        return math.prod(os)

    def get_gemms(self):
        # TODO might have some problems
        return 0, 0, 0, 1


class UpsampleBilinear2d(Operator):          ## Vector Execution Unit
    def __init__(self, node, density=(1.0, 1.0, 1.0)):
        super().__init__(node, density)
        self.alu_type = 'vxu'

    def get_tensors(self):
        # input and output
        return self.node.inputs[0].shape, self.node.outputs[0].shape

    def get_num_ops(self):
        os = self.node.outputs[0].shape
        return math.prod(os) * 4

    def get_gemms(self):
        # TODO might have some problems
        return 0, 0, 0, 1


operator_list = {
    "aten::addmm": Addmm,
    "aten::addmv": Addmv,
    "aten::bmm": Bmm,
    "aten::baddbmm": Baddbmm,
    "aten::linear": Matmul,
    "aten::matmul": Matmul,
    "aten::reciprocal": Special_Func,
    "aten::mul": Mul,
    "aten::mul_": Mul,
    "aten::_convolution": Convolution,
    "aten::batch_norm": Norm,
    "aten::instance_norm": Norm,
    "aten::layer_norm": Norm,
    "aten::group_norm": Norm,
    "aten::adaptive_avg_pool1d": Avg,
    "aten::adaptive_avg_pool2d": Avg,
    "aten::adaptive_avg_pool3d": Avg,
    "aten::avg_pool1d": Avg,
    "aten::avg_pool2d": Avg,
    "aten::avg_pool3d": Avg,
    "aten::mean": Avg,
    "aten::leaky_relu": Special_Func,
    "aten::upsample_bilinear2d": UpsampleBilinear2d,
    "aten::cos": Special_Func,
    "aten::sin": Special_Func,
    "aten::adaptive_max_pool1d": None,
    "aten::adaptive_max_pool2d": None,
    "aten::adaptive_max_pool3d": None,
    "aten::add": None,
    "aten::add_": None,
    "aten::alpha_dropout": None,
    "aten::cat": None,
    "aten::chunk": None,
    "aten::clamp": None,
    "aten::clone": None,
    "aten::constant_pad_nd": None,
    "aten::contiguous": None,
    "aten::detach": None,
    "aten::div": None,
    "aten::div_": None,
    "aten::dropout": None,
    "aten::dropout_": None,
    "aten::embedding": None,
    "aten::eq": None,
    "aten::feature_dropout": None,
    "aten::flatten": None,
    "aten::floor": None,
    "aten::floor_divide": None,
    "aten::gt": None,
    "aten::hardtanh_": None,
    "aten::hardtanh": None,
    "aten::index": None,
    "aten::int": None,
    "aten::log_softmax": Special_Func,
    "aten::lt": None,
    "aten::max_pool1d": None,
    "aten::max_pool1d_with_indices": None,
    "aten::max_pool2d": None,
    "aten::max_pool2d_with_indices": None,
    "aten::max_pool3d": None,
    "aten::max_pool3d_with_indices": None,
    "aten::max_unpool1d": None,
    "aten::max_unpool2d": None,
    "aten::max_unpool3d": None,
    "aten::ne": None,
    "aten::reflection_pad1d": None,
    "aten::reflection_pad2d": None,
    "aten::reflection_pad3d": None,
    "aten::relu": Special_Func,
    "aten::relu_": Special_Func,
    "aten::gelu": Special_Func,
    "aten::replication_pad1d": None,
    "aten::replication_pad2d": None,
    "aten::replication_pad3d": None,
    "aten::rsub": None,
    "aten::select": None,
    "aten::sigmoid": Special_Func,
    "aten::size": None,
    "aten::slice": None,
    "aten::softmax": Special_Func,
    "aten::softshrink": None,
    "aten::squeeze": None,
    "aten::stack": None,
    "aten::sub": None,
    "aten::sum": None,
    "aten::t": None,
    "aten::tanh": Special_Func,
    "aten::threshold": None,
    "aten::to": None,
    "aten::transpose": None,
    "aten::upsample_nearest2d": None,
    "aten::view": None,
    "aten::zeros": None,
    "prim::constant": None,
    "prim::listconstruct": None,
    "prim::listunpack": None,
    "prim::numtotensor": None,
    "prim::tupleconstruct": None,
    "aten::scalarimplicit": None,
    "aten::scaled_dot_product_attention": ScaledDotProductAttention,
    "aten::arange": None,
    "aten::unsqueeze": None,
    "aten::repeat": None,
    "aten::pow":  Special_Func,
    "aten::rsqrt": Special_Func,
    "aten::type_as": None,
    "aten::expand": None,
    "aten::neg": Special_Func, 
}
