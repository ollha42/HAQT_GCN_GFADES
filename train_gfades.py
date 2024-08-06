import numpy as np
import torch
import torch.nn.functional as F
import sys
import math
import pandas as pd
import time

np.set_printoptions(linewidth=200, precision=4, suppress=True)

from torch_geometric.nn import GCNConv
from torch.nn import Linear, Dropout
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

from pynq import allocate
from pynq import Overlay

########################################################################################################################

def accuracy(x, labels):

    x1 = np.equal(x, labels)
    x2 = np.sum(x1)

    if isinstance(x, list):
        acc = x2 / len(x)
    else:
        acc = x2 / x.size
    return acc

def float_to_fix(array, cmin, cmax, frac, bits, sign=1):
    if sign == 0:
        if bits > 8:
            tmp = np.clip(array, cmin, cmax)
            return np.round(tmp * (2**frac)).astype(np.uint16)
        else:
            tmp = np.clip(array, cmin, cmax)
            return np.round(tmp * (2**frac)).astype(np.uint8)
    else:
        if bits > 16:
            tmp = np.clip(array, cmin, cmax)
            return np.round(tmp * (2**frac)).astype(np.int32)
        elif bits > 8:
            tmp = np.clip(array, cmin, cmax)
            return np.round(tmp * (2**frac)).astype(np.int16)
        else:
            tmp = np.clip(array, cmin, cmax)
            return np.round(tmp * (2**frac)).astype(np.int8)
        
def float_to_fix_reduced(array, cmin, cmax, frac, bits, sign=1, reduced=0):
    if sign == 0:
        if bits > 8:
            tmp = np.clip(array, cmin, cmax)
            return (np.round(tmp * (2**(frac-reduced)))*(2**reduced)).astype(np.uint16)
        else:
            tmp = np.clip(array, cmin, cmax)
            return (np.round(tmp * (2**(frac-reduced)))*(2**reduced)).astype(np.uint8)
    else:
        if bits > 16:
            tmp = np.clip(array, cmin, cmax)
            return (np.round(tmp * (2**(frac-reduced)))*(2**reduced)).astype(np.int32)
        elif bits > 8:
            tmp = np.clip(array, cmin, cmax)
            return (np.round(tmp * (2**(frac-reduced)))*(2**reduced)).astype(np.int16)
        else:
            tmp = np.clip(array, cmin, cmax)
            return (np.round(tmp * (2**(frac-reduced)))*(2**reduced)).astype(np.int8)


def fix_to_float(array, frac, bits):

    if bits < 8: # handle sign if not correct number of bits, eg 4 in 8 or 12 in 16
        shift_length = 8 - bits
        tmp = np.left_shift(array, shift_length)
        tmp2 = np.right_shift(tmp, shift_length).astype(np.float32)
    elif bits == 12:
        shift_length = 16 - bits
        tmp = np.left_shift(array, shift_length)
        tmp2 = np.right_shift(tmp, shift_length).astype(np.float32)
    else:
        tmp2 = array.astype(np.float32)
    
    res = (tmp2 / (2**frac)).astype(np.float32)
    return res


def generate_quantization_constants(x, percentile=100, beta_q=1, sign=1):
            if sign == 0:
                beta = np.percentile(x, percentile)
                s = beta / beta_q
                return s
            else:
                beta = np.percentile(x, percentile)
                alpha = np.min(x)
                sign = -beta_q if alpha < 0 else 0
                s = (beta - alpha) / (beta_q - sign)
                return s


def scale(x, s, cmin=0, cmax=1):
    return np.clip(x / s, cmin, cmax)


def descale(x, s):
        return x * s

########################################################################################################################

class GCN_ref(torch.nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels
    ):
        super(GCN_ref, self).__init__()
        torch.manual_seed(12345)

        self.conv1 = GCNConv(in_channels, hidden_channels, bias=False)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, bias=False)
        self.lin1 = Linear(hidden_channels, out_channels)

        self.dropout = Dropout(0.5)

    def forward(self, x, adj_dense):   

        x = self.conv1(x, adj_dense)
        x = x.relu()

        x = self.conv2(x, adj_dense)
        x = x.relu()

        x = self.dropout(x)
        x = self.lin1(x)

        return x
    
########################################################################################################################

class RPYNQ(torch.autograd.Function):
    """Both forward and backward are static methods."""

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        ctx.save_for_backward(input)

        output = input.clone()

        return output

    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the inputs: here input and weights
        """
        (input,) = ctx.saved_tensors

        grad_input = grad_output.clone()

        grad_input[input == 0] = 0

        return grad_input


class FPYNQ(torch.autograd.Function):
    """Both forward and backward are static methods."""

    @staticmethod
    def forward(ctx, my_ip, adj, input, weights, name, adj_s, fea_s):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        l = 0
        if name == "conv2":
            l = 1

        if acc == 1:

            my_ip.register_map.N_adj = adj.shape[0]
            my_ip.register_map.M_adj = adj.shape[1]
            my_ip.register_map.M_fea = input.shape[1]
            my_ip.register_map.P_w = weights.shape[1]     

            support = weights.t()
            support_pynq = support.data.numpy()
            w_s = 1
            if do_wscale == 1:
                max_val = ((2**(float(wbits)-float(wfrac)-1))-(2**(-float(wfrac))))
                #max_val = ((2**(float(wbits)-float(wfrac)-1))-(2**(-(float(wfrac)-float(reduced)))))
                w_s = generate_quantization_constants(support_pynq, percentile, max_val)
                support_pynq = scale(support_pynq, w_s, -max_val, max_val)
            support_pynq_q = support_pynq.reshape(1, (weights.shape[0]*weights.shape[1]))
            if fix == 1:
                B_buffer[0 : (weights.shape[0] * weights.shape[1])] = float_to_fix(support_pynq_q, wmin, wmax, wfrac, wbits, 1)
                #B_buffer[0 : (weights.shape[0] * weights.shape[1])] = float_to_fix_reduced(support_pynq_q, wmin, wmax, wfrac, wbits, 1, reduced)
            else:
                B_buffer[0 : (weights.shape[0] * weights.shape[1])] = support_pynq_q.astype(acc_datatype)

            # tik
            tik = time.time()

            my_ip.register_map.CTRL.AP_START = 1
            kernel_done = my_ip.register_map.CTRL.AP_DONE
            while kernel_done == 0:
                kernel_done = my_ip.register_map.CTRL.AP_DONE
            
            # tok
            tok = time.time()
            times[epoch, 0, l] = tok - tik

            if fix == 1:
                output_acc = fix_to_float(D_buffer[0 : adj.shape[0] * weights.shape[1]], dfrac, dbits)
            else:
                output_acc = D_buffer[0 : adj.shape[0] * weights.shape[1]].astype(np.float32)

            output_acc = output_acc.reshape(adj.shape[0], weights.shape[1])

            if do_scale == 1:
                output_acc = descale(output_acc, adj_s*fea_s*w_s)

            output_acc = torch.from_numpy(output_acc)

            
            ctx.save_for_backward(adj, input, weights)
            ctx.name = name

            return output_acc
        else:

            # tik
            tik = time.time()

            support = torch.mm(input, weights)
            output_cpu = torch.spmm(adj, support)

            # tok
            tok = time.time()
            times[epoch, 0, l] = tok - tik

            ctx.save_for_backward(adj, input, weights)
            ctx.name = name

            return output_cpu

    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the inputs: here input and weights
        """

        l = 0
        if ctx.name == "conv2":
            l = 1

        if accb == 1:

            adj, input, weights = ctx.saved_tensors

            my_ip.register_map.gemm_mode = 1
            my_ip.register_map.relu = 0

            len_val = len(adj.crow_indices())
            rowPtr_adj_buffer[0:len_val] = adj.crow_indices()

            len_val = len(adj.col_indices())
            columnIndex_adj_buffer[0:len_val] = adj.col_indices()

            len_val = len(adj.values())
            adj_np = adj.values().numpy()
            adj_s1 = 1
            if do_ascale == 1:
                max_val = ((2**(float(abits)-float(afrac)))-(2**(-float(afrac))))
                adj_s1 = generate_quantization_constants(adj_np, percentile, max_val, 0)
                adj_np = scale(adj_np, adj_s1, 0, max_val)
            if fix == 1:
                values_adj_buffer[0:len_val] = float_to_fix(adj_np, amin, amax, afrac, abits, asign)
            else:
                values_adj_buffer[0:len_val] = adj_np.astype(acc_datatype)

            support_pynq_b = grad_output.data.numpy()
            go_s1 = 1
            if do_gscale == 1:
                max_val = ((2**(float(fbits)-float(ffrac)-1))-(2**(-float(ffrac))))
                go_s1 = generate_quantization_constants(support_pynq_b, percentile, max_val)
                support_pynq_b = scale(support_pynq_b, go_s1, -max_val, max_val)
            support_pynq_b = support_pynq_b.reshape(1, (grad_output.shape[0] * grad_output.shape[1]))
            if fix == 1:
                values_fea_buffer[0 : (grad_output.shape[0] * grad_output.shape[1])] = float_to_fix(support_pynq_b, fmin, fmax, ffrac, fbits, fsign)
            else:
                values_fea_buffer[0 : (grad_output.shape[0] * grad_output.shape[1])] = support_pynq_b.astype(acc_datatype)

            support_pynq_g = weights.data.numpy()
            w_s1 = 1
            if do_wscale == 1:
                max_val = ((2**(float(wbits)-float(wfrac)-1))-(2**(-float(wfrac)))) 
                w_s1 = generate_quantization_constants(support_pynq_g, percentile, max_val)
                support_pynq_g = scale(support_pynq_g, w_s1, -max_val, max_val)
            support_pynq_g = support_pynq_g.reshape(1, (weights.shape[0] * weights.shape[1]))
            if fix == 1:
                B_buffer[0 : (weights.shape[0] * weights.shape[1])] = float_to_fix(support_pynq_g, wmin, wmax, wfrac, wbits, 1)
            else:
                B_buffer[0 : (weights.shape[0] * weights.shape[1])] = support_pynq_g.astype(acc_datatype)

            my_ip.register_map.N_adj = adj.shape[0]
            my_ip.register_map.M_adj = grad_output.shape[0]
            my_ip.register_map.M_fea = grad_output.shape[1]
            my_ip.register_map.P_w = weights.shape[0]

            # tik
            tik = time.time()
            
            my_ip.register_map.CTRL.AP_START = 1
            kernel_done = my_ip.register_map.CTRL.AP_DONE
            while kernel_done == 0:
                kernel_done = my_ip.register_map.CTRL.AP_DONE

            # tok
            tok = time.time()
            times[epoch, 1, l] = tok - tik

            if fix == 1:
                output_acc2 = fix_to_float(D_buffer[0 : adj.shape[0] * weights.shape[0]], dfrac, dbits)
            else:
                output_acc2 = D_buffer[0 : adj.shape[0] * weights.shape[0]].astype(np.float32)

            if do_scale == 1:
                output_acc2 = descale(output_acc2, adj_s1*w_s1*go_s1)
            output_acc2 = output_acc2.reshape(adj.shape[0], weights.shape[0])
            grad_input = torch.from_numpy(output_acc2).clone()
            grad_input = grad_input.float()

            #

            my_ip.register_map.gemm_mode = 2
            my_ip.register_map.relu = 0


            input_t = input.t()
            support_pynq_b = input_t.data.numpy()
            fea_s2 = 1
            support_pynq_b_nz = support_pynq_b[np.nonzero(support_pynq_b)]
            if do_fscale == 1:
                max_val = ((2**(float(abits)-float(afrac)))-(2**(-float(afrac))))
                fea_s2 = generate_quantization_constants(support_pynq_b_nz, percentile, max_val, 0)
                support_pynq_b_nz = scale(support_pynq_b_nz, fea_s2, 0, max_val)
                support_pynq_b[np.nonzero(support_pynq_b)] = support_pynq_b_nz
            support_pynq_b = support_pynq_b.reshape(1, (input_t.shape[0] * input_t.shape[1]))
            if fix == 1:                
                values_adj_buffer[0 : (input_t.shape[0] * input_t.shape[1])] = float_to_fix(support_pynq_b, amin, amax, afrac, abits, asign)
            else:
                values_adj_buffer[0 : (input_t.shape[0] * input_t.shape[1])] = support_pynq_b.astype(acc_datatype)


            len_val = len(adj.crow_indices())
            rowPtr_fea_buffer[0:len_val] = adj.crow_indices()

            len_val = len(adj.col_indices())
            columnIndex_fea_buffer[0:len_val] = adj.col_indices()

            len_val = len(adj.values())
            adj_np = adj.values().numpy()
            adj_s2 = 1
            if do_ascale == 1:
                max_val = ((2**(float(fbits)-float(ffrac)-1))-(2**(-float(ffrac))))
                adj_s2 = generate_quantization_constants(adj_np, percentile, max_val, 0)
                adj_np = scale(adj_np, adj_s2, 0, max_val)
            if fix == 1:
                values_fea_buffer[0:len_val] = float_to_fix(adj_np, fmin, fmax, ffrac, fbits, fsign)
            else:
                values_fea_buffer[0:len_val] = adj_np.astype(acc_datatype)


            support_g = grad_output.t()
            support_pynq_g = support_g.data.numpy()
            go_s2 = 1
            if do_gscale == 1:
                max_val = ((2**(float(wbits)-float(wfrac)-1))-(2**(-float(wfrac))))
                go_s2 = generate_quantization_constants(support_pynq_g, percentile, max_val)
                support_pynq_g = scale(support_pynq_g, go_s2, -max_val, max_val)
            support_pynq_g = support_pynq_g.reshape(1, (grad_output.shape[0] * grad_output.shape[1]))
            if fix == 1:
                B_buffer[0 : (grad_output.shape[0] * grad_output.shape[1])] = float_to_fix(support_pynq_g, wmin, wmax, wfrac, wbits, 1)
            else:
                B_buffer[0 : (grad_output.shape[0] * grad_output.shape[1])] = support_pynq_g.astype(acc_datatype)

            my_ip.register_map.N_adj = input_t.shape[0]
            my_ip.register_map.M_adj = adj.shape[0]
            my_ip.register_map.M_fea = adj.shape[1]
            my_ip.register_map.P_w = grad_output.shape[1]
            
            # tik
            tik = time.time()

            my_ip.register_map.CTRL.AP_START = 1
            kernel_done = my_ip.register_map.CTRL.AP_DONE
            while kernel_done == 0:
                kernel_done = my_ip.register_map.CTRL.AP_DONE

            # tok
            tok = time.time()
            times[epoch, 2, l] = tok - tik

            if fix == 1:
                output_acc = fix_to_float(D_buffer[0 : input_t.shape[0] * grad_output.shape[1]], dfrac, dbits)
            else:
                output_acc = D_buffer[0 : input_t.shape[0] * grad_output.shape[1]].astype(np.float32)

            if do_scale == 1:
                output_acc = descale(output_acc, adj_s2*fea_s2*go_s2)
            output_acc = output_acc.reshape(input_t.shape[0], grad_output.shape[1])
            grad_weights = torch.from_numpy(output_acc).clone()
            grad_weights = grad_weights.float()

        else:

            adj, input, weights = ctx.saved_tensors

            # 1 gI
            weights_t = weights.t()
            # tik
            tik = time.time()
            support = torch.mm(grad_output, weights_t)
            grad_input = torch.spmm(adj, support)
            # tok
            tok = time.time()
            times[epoch, 1, l] = tok - tik

            # 2 gW
            input_t = input.t()
            # tik
            tik = time.time()
            support = torch.spmm(adj, grad_output)
            grad_weights = torch.mm(input_t, support)
            # tok
            tok = time.time()
            times[epoch, 2, l] = tok - tik
            

        return None, None, grad_input, grad_weights, None, None, None

class Relu_pynq(Module):
    """
    Relu activation.

    The forward pass receives the input data (array) and exchanges any negative
    entry for zero.

    The backward pass should calculate the gradient of the maximum function in
    the forward pass and return it
    """

    def __init__(self):
        super(Relu_pynq, self).__init__()
        self.fn = RPYNQ.apply

    def forward(self, x):
        output = self.fn(x)
        return output

class GraphConvolution_pynq(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, my_ip, name=None):
        super(GraphConvolution_pynq, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.FloatTensor(in_features, out_features))

        self.fn = FPYNQ.apply
        self.my_ip = my_ip
        self.name = name
        self.reset_parameters()

    def reset_parameters(self):
        #stdv = 1.0 / math.sqrt(self.weight.size(1))
        stdv = math.sqrt(6.0 / (self.weight.size(-2) + self.weight.size(-1)))
        self.weight.data.uniform_(-stdv, stdv)

    def run_kernel(self):
        self.my_ip.register_map.CTRL.AP_START = 1
        kernel_done = self.my_ip.register_map.CTRL.AP_DONE
        while kernel_done == 0:
            kernel_done = self.my_ip.register_map.CTRL.AP_DONE

    def forward(
        self,
        dense,
        relu,
        input,
        adj,
        adj_s = 1,
        fea_s = 1
    ):

        self.my_ip.register_map.relu = relu
        self.my_ip.register_map.gemm_mode = dense
        output_acc = self.fn(self.my_ip, adj, input, self.weight, self.name, adj_s, fea_s)
        output = output_acc

        return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )
    
class GCN_PYNQ(torch.nn.Module):

    def __init__(self, hidden_channels, num_features, num_classes):
        super(GCN_PYNQ, self).__init__()
        torch.manual_seed(12345)

        self.conv1 = GraphConvolution_pynq(num_features, hidden_channels, my_ip, "conv1")
        self.conv2 = GraphConvolution_pynq(hidden_channels, hidden_channels, my_ip, "conv2")

        self.reluh = Relu_pynq()

        self.dropout = Dropout(0.5)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(
        self,
        acc,
        x,
        adj,
        rowPtr_fea_buffer,
        columnIndex_fea_buffer,
        values_fea_buffer,
        rowPtr_adj_buffer,
        columnIndex_adj_buffer,
        values_adj_buffer,
    ):
        
        dense = 0
        adj_s = 1
        fea_s = 1

        if acc == 1:
            len_val = len(adj.crow_indices())
            rowPtr_adj_buffer[0:len_val] = adj.crow_indices()

            len_val = len(adj.col_indices())
            columnIndex_adj_buffer[0:len_val] = adj.col_indices()

            len_val = len(adj.values())
            adj_np = adj.values().numpy()
            adj_s = 1
            if do_ascale == 1:
                max_val = ((2**(float(abits)-float(afrac)))-(2**(-float(afrac))))
                #max_val = ((2**(float(abits)-float(afrac)))-(2**(-(float(afrac)-float(reduced)))
                adj_s = generate_quantization_constants(adj_np, percentile, max_val, 0)
                adj_np = scale(adj_np, adj_s, 0, max_val)
            if fix == 1:
                values_adj_buffer[0:len_val] = float_to_fix(adj_np, amin, amax, afrac, abits, asign)
                #values_adj_buffer[0:len_val] = float_to_fix_reduced(adj_np, amin, amax, afrac, abits, asign, reduced)
            else:
                values_adj_buffer[0:len_val] = adj_np.astype(acc_datatype)


            support_x = x
            pynq_features = support_x._to_sparse_csr()

            len_val = len(pynq_features.crow_indices())
            rowPtr_fea_buffer[0:len_val] = pynq_features.crow_indices()

            len_val = len(pynq_features.col_indices())
            columnIndex_fea_buffer[0:len_val] = pynq_features.col_indices()

            len_val = len(pynq_features.values())
            fea_np = pynq_features.values().numpy()
            fea_s = 1
            if do_fscale == 1:
                max_val = ((2**(float(fbits)-float(ffrac)-1))-(2**(-float(ffrac))))
                #max_val = ((2**(float(fbits)-float(ffrac)-1))-(2**(-(float(ffrac)-float(reduced))))
                fea_s = generate_quantization_constants(fea_np, percentile, max_val, 0)
                fea_np = scale(fea_np, fea_s, 0, max_val)
            if fix == 1:
                values_fea_buffer[0:len_val] = float_to_fix(fea_np, fmin, fmax, ffrac, fbits, fsign)
                #values_fea_buffer[0:len_val] = float_to_fix_reduced(fea_np, fmin, fmax, ffrac, fbits, fsign, reduced)
            else:
                values_fea_buffer[0:len_val] = fea_np.astype(acc_datatype)
        
        relu = 1

        x = self.conv1(
            dense,
            relu,
            x,
            adj,
            fea_s
        )

        x = x.float()

        if acc == 0:
            x = x.relu()
        else:
            x = self.reluh(x)

        dense = 1

        if acc == 1:
            xaux = x.detach().numpy()
            fea_s = 1
            xaux_nz = xaux[np.nonzero(xaux)]
            if do_fscale == 1:
                max_val = ((2**(float(fbits)-float(ffrac)-1))-(2**(-float(ffrac))))
                #max_val = ((2**(float(fbits)-float(ffrac)-1))-(2**(-(float(ffrac)-float(reduced)))
                fea_s = generate_quantization_constants(xaux_nz, percentile, max_val, 0)
                xaux_nz = scale(xaux_nz, fea_s, 0, max_val)
                xaux[np.nonzero(xaux)] = xaux_nz
            if fix == 1:
                values_fea_buffer[0 : (x.shape[0] * x.shape[1])] = float_to_fix(xaux.reshape(1, x.shape[0] * x.shape[1]), fmin, fmax, ffrac, fbits, fsign)
                #values_fea_buffer[0 : (x.shape[0] * x.shape[1])] = float_to_fix_reduced(xaux.reshape(1, x.shape[0] * x.shape[1]), fmin, fmax, ffrac, fbits, fsign, reduced)
            else:
                values_fea_buffer[0 : (x.shape[0] * x.shape[1])] = xaux.reshape(1, x.shape[0] * x.shape[1]).astype(acc_datatype)

        relu = 1

        x = self.conv2(
            dense,
            relu,
            x,
            adj,
            fea_s
        )

        x = x.float()

        if acc == 0:
            x = x.relu()
        else:
            x = self.reluh(x)

        x = self.dropout(x)
        x = self.lin(x)

        return x

########################################################################################################################

def train(train_loader):
    model.train()

    total_loss = 0
    n_entries = 0

    for data in train_loader:
        if custom:
            adj_t = data.adj_t.to(device)
            x, y, train_mask = (data.x.to(device),data.y.to(device),data.train_mask.to(device))
        else:
            x, edge_index, y, train_mask = (data.x.to(device),data.edge_index.to(device),data.y.to(device),data.train_mask.to(device))

        optimizer.zero_grad()

        if custom:
            z = model(
                acc,
                x,
                adj_t,
                rowPtr_fea_buffer,
                columnIndex_fea_buffer,
                values_fea_buffer,
                rowPtr_adj_buffer,
                columnIndex_adj_buffer,
                values_adj_buffer,
            )   
        else:
            z = model(x, edge_index)
            
        loss = criterion(z[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
        
        total_loss += loss.detach().cpu().numpy() * np.count_nonzero(train_mask.detach().cpu().numpy())
        n_entries += np.count_nonzero(train_mask.detach().cpu().numpy())

    return total_loss / n_entries


def eval(test_loader):
    model.eval()

    total_loss = 0
    n_entries = 0
    test_pred, test_true = [], []

    for data in test_loader:

        if custom:
            adj_t = data.adj_t.to(device)
            x, y, test_mask = (data.x.to(device),data.y.to(device),data.test_mask.to(device))
        else:
            x, edge_index, y, test_mask = (data.x.to(device),data.edge_index.to(device),data.y.to(device),data.test_mask.to(device))

        with torch.no_grad():
            if custom:
                z = model(
                    acc,
                    x,
                    adj_t,
                    rowPtr_fea_buffer,
                    columnIndex_fea_buffer,
                    values_fea_buffer,
                    rowPtr_adj_buffer,
                    columnIndex_adj_buffer,
                    values_adj_buffer,
                )   
            else:
                z = model(x, edge_index)

            loss = criterion(z[test_mask], y[test_mask])
            test_pred.append(z[test_mask].detach().cpu().numpy())
            test_true.append(y[test_mask].detach().cpu().numpy())

            
            total_loss += loss.detach().cpu().numpy() * np.count_nonzero(test_mask.detach().cpu().numpy())
            n_entries += np.count_nonzero(test_mask.detach().cpu().numpy())

    return total_loss / n_entries, np.vstack(test_pred), np.concatenate(test_true)

########################################################################################################################


seed = 12345
np.random.seed(12345)
torch.manual_seed(12345)
torch.cuda.manual_seed(12345)
np.set_printoptions(linewidth=200, precision=4, suppress=True, edgeitems=8)
torch.set_printoptions(linewidth=200, precision=4, edgeitems=8)


design_config = None
dataset_sel = None
backward_config = None
scale_config = None
results_dir = None
reduced = 0
if len(sys.argv) > 1:
    design_config = sys.argv[1]
    print(f"design_config: {design_config}")
    if len(sys.argv) > 2:
        dataset_sel = sys.argv[2]
        print(f"dataset_sel: {dataset_sel}")
        if len(sys.argv) > 3:
            backward_config = sys.argv[3]
            print(f"backward_config: {backward_config}")
            if len(sys.argv) > 4:
                scale_config = int(sys.argv[4])
                print(f"scale_config: {scale_config}")
                if len(sys.argv) > 5:
                    results_dir = sys.argv[5]
                    print(f"results_dir: {results_dir}")
                    if len(sys.argv) > 6:
                        reduced = int(sys.argv[6])
                        print(f"reduced: {reduced}")


custom = 1
if design_config == "cpu":
    accb = 0
else:
    if backward_config == None:
        accb = 1
    else:
        accb = int(backward_config)
if (design_config == "cpu"):
    acc = 0
else:
    acc = 1

if accb == 1:
    print("Accelerated backward")
else:
    print("CPU backward")

#internal_scale = 0x80000000
internal_scale = 1
#internal_quantization = 0x7f000000
#internal_quantization = 127
internal_quantization = 64

if scale_config == None:
    do_ascale = 1 # adjacency scale
    do_fscale = 1 # feature scale
    do_wscale = 1 # weight scale
    do_gscale = 1 # gradient scale
else:
    do_ascale = scale_config
    do_fscale = scale_config
    do_wscale = scale_config
    do_gscale = scale_config
do_scale = do_ascale or do_fscale or do_wscale or do_gscale
percentile = 99.9

learning_rate = 1e-2

batch_size = 64
hidden_channels = 64
nrof_epochs = 64

# Cora, CiteSeer, PubMed
if dataset_sel == None:
    dataset_sel = "Cora"

if dataset_sel == "Cora":
    weight_decay = 1e-2
elif dataset_sel == "CiteSeer":
    weight_decay = 2e-2
elif dataset_sel == "PubMed":
    weight_decay = 1e-3

P_w = hidden_channels  # nhid number hidden units   in first layer
if dataset_sel == "Cora":
    N_adj = 2708  
    M_adj = 2708
    M_fea = 1433
    NNZ_adj = 13264
    NNZ_fea = 49216
elif dataset_sel == "CiteSeer":
    N_adj = 3327
    M_adj = 3327
    M_fea = 3703
    NNZ_adj = 12431
    NNZ_fea = 105165
elif dataset_sel == "PubMed":
    N_adj = 19717
    M_adj = 19717
    M_fea = 500
    NNZ_adj = 108365
    NNZ_fea = 988031
else:
    N_adj = 7000  # 6144  # number of nodes
    M_adj = 7000  # 6144  # number of nodes
    M_fea = 7000  # max number of input features
    NNZ_adj = 500000  # 12431 # number of non-zero values of adjacency
    NNZ_fea = 500000  # 105165 # number of non-zero values of feature

if design_config == "328":
    design_config_split = [32,8,32,8,32,8,32,8,32,8]
elif (design_config != None):
    design_config_split = design_config.split("_")

if (design_config == "float") or (design_config == "half") or (design_config == "cpu"):
    fix = 0
else:
    fix = 1

qbits = None
if fix == 1:
    if design_config == None:
        abits = 4
    else:
        abits = int(design_config_split[0])
    if design_config == None:
        aint = -1
    else:
        aint = int(design_config_split[1])
    if design_config == "328":
        asign = 1
    else:
        asign = 0
    afrac = abits - aint
    if asign == 1:
        amin = -(2 ** (aint - 1))
        amax = (2 ** (aint - 1)) - (2 ** (-afrac))
    else:
        amin = 0
        amax = (2 ** (aint)) - (2 ** (-afrac))

    if design_config == None:
        fbits = 4
    else:
        fbits = int(design_config_split[2])
    if design_config == None:
        fint = 0
    else:
        fint = int(design_config_split[3])
    fsign = 1
    ffrac = fbits - fint
    if fsign == 1:
        fmin = -(2 ** (fint - 1))
        fmax = (2 ** (fint - 1)) - (2 ** (-ffrac))
    else:
        fmin = 0
        fmax = (2 ** (fint)) - (2 ** (-ffrac))

    if design_config == None:
        wbits = 4
    else:
        wbits = int(design_config_split[4])
    if design_config == None:
        wint = 0
    else:
        wint = int(design_config_split[5])
    wfrac = wbits - wint
    wmin = -(2 ** (wint - 1))
    wmax = (2 ** (wint - 1)) - (2 ** (-wfrac))

    if design_config == None:
        qbits = 10
    else:
        qbits = int(design_config_split[6])
    if design_config == None:
        qint = 2
    else:
        qint = int(design_config_split[7])

    if design_config == None:
        dbits = 16
    else:
        dbits = int(design_config_split[8])
    if design_config == None:
        dint = 3
    else:
        dint = int(design_config_split[9])
    dfrac = dbits - dint
    dmin = -(2 ** (dint - 1))
    dmax = (2 ** (dint - 1)) - (2 ** (-dfrac))

    ibits = 32
    iint = 8
    ifrac = ibits - iint
    imin = -(2 ** (wint - 1))
    imax = (2 ** (wint - 1)) - (2 ** (-wfrac))

    if abits > 16:
        values_adj_buffer = allocate(max(NNZ_adj, M_fea*N_adj, P_w*N_adj), dtype=np.int32)
        #values_adj_buffer = allocate(NNZ_adj, dtype=np.int32)
    elif abits > 8:
        if asign == 1:
            values_adj_buffer = allocate(NNZ_adj, dtype=np.int16)
        else:
            values_adj_buffer = allocate(NNZ_adj, dtype=np.uint16)
    else:
        if asign == 1:
            values_adj_buffer = allocate(NNZ_adj, dtype=np.int8)
        else:
            values_adj_buffer = allocate(max(NNZ_adj, M_fea*N_adj, P_w*N_adj), dtype=np.uint8)
            #values_adj_buffer = allocate(NNZ_adj, dtype=np.uint8)

    if fbits > 16:
        values_fea_buffer = allocate(max(NNZ_fea, N_adj*P_w, NNZ_adj), dtype=np.int32)
        #values_fea_buffer = allocate(max(NNZ_fea, N_adj*P_w, M_fea*N_adj), dtype=np.int32)
    elif fbits > 8:
        if fsign == 1:
            values_fea_buffer = allocate(max(NNZ_fea, N_adj*P_w, M_fea*N_adj), dtype=np.int16)
        else:
            values_fea_buffer = allocate(max(NNZ_fea, N_adj*P_w, M_fea*N_adj), dtype=np.uint16)
    else:
        if fsign == 1:
            values_fea_buffer = allocate(max(NNZ_fea, N_adj*P_w, NNZ_adj), dtype=np.int8)
            #values_fea_buffer = allocate(max(NNZ_fea, N_adj*P_w, M_fea*N_adj), dtype=np.int8)
        else:
            values_fea_buffer = allocate(max(NNZ_fea, N_adj*P_w, M_fea*N_adj), dtype=np.uint8)

    if wbits > 16:
        B_buffer = allocate(max(M_fea*P_w, P_w * P_w, N_adj*P_w), dtype=np.int32)
    elif wbits > 8:
        B_buffer = allocate(max(M_fea*P_w, P_w * P_w, N_adj*P_w), dtype=np.int16)
    else:
        B_buffer = allocate(max(M_fea*P_w, P_w * P_w, N_adj*P_w), dtype=np.int8)

    if dbits > 16:
        D_buffer = allocate(max((N_adj * P_w),(N_adj*M_fea),(M_fea*P_w),(P_w*P_w)), dtype=np.int32)
    elif dbits > 8:
        D_buffer = allocate(max((N_adj * P_w),(N_adj*M_fea),(M_fea*P_w),(P_w*P_w)), dtype=np.int16)
    else:
        D_buffer = allocate(max((N_adj * P_w),(N_adj*M_fea),(M_fea*P_w),(P_w*P_w)), dtype=np.int8)
else:
    ol = Overlay("designs/design_b_float_224.bit")
    values_adj_buffer = allocate(max(NNZ_adj, M_fea*N_adj, P_w*N_adj), dtype=np.float32)
    values_fea_buffer = allocate(max(NNZ_fea, N_adj*P_w, NNZ_adj), dtype=np.float32)
    B_buffer = allocate(max(M_fea*P_w, P_w * P_w, N_adj*P_w), dtype=np.float32)
    D_buffer = allocate(max((N_adj * P_w),(N_adj*M_fea),(M_fea*P_w),(P_w*P_w)), dtype=np.float32)
    acc_datatype = np.float32

if design_config == None:
    #design = "design_all_328b_222"
    design = "design_b_au4-1f40b40q102d163_448"
    #design = "design_b_au8-1f80b80q204d327_444"
    #design = "design_b_float_224"
elif design_config == "cpu":
    design = "design_b_float_224"
elif design_config == "328":
    design = "design_all_328b_222"
elif design_config == "float":
    design = "design_b_float_224"
else:
    if qbits > 11:
        para_config = "444"
    else:
        para_config = "448"
    design = "design_b_" + "au" + str(abits) + str(aint) + "f" + str(fbits) + str(fint) + "b" + str(wbits) + str(wint) + "q" + str(qbits) + str(qint) + "d" + str(dbits) + str(dint) + "_" + para_config
    
print(f"Design: {design}")

ol = Overlay("designs/" + design + ".bit")

my_ip = ol.mmult_top_0

rowPtr_fea_buffer = allocate(N_adj + 1, dtype=np.int32)
columnIndex_fea_buffer = allocate(NNZ_fea, dtype=np.int32)
rowPtr_adj_buffer = allocate(N_adj + 1, dtype=np.int32)
columnIndex_adj_buffer = allocate(NNZ_adj, dtype=np.int32)

my_ip.register_map.B_offset_1 = B_buffer.physical_address
my_ip.register_map.rowPtr_fea1_offset_1 = rowPtr_fea_buffer.physical_address
my_ip.register_map.rowPtr_fea2_offset_1 = rowPtr_fea_buffer.physical_address
my_ip.register_map.rowPtr_fea3_offset_1 = rowPtr_fea_buffer.physical_address
my_ip.register_map.rowPtr_fea4_offset_1 = rowPtr_fea_buffer.physical_address
my_ip.register_map.columnIndex_fea1_offset_1 = columnIndex_fea_buffer.physical_address
my_ip.register_map.columnIndex_fea2_offset_1 = columnIndex_fea_buffer.physical_address
my_ip.register_map.columnIndex_fea3_offset_1 = columnIndex_fea_buffer.physical_address
my_ip.register_map.columnIndex_fea4_offset_1 = columnIndex_fea_buffer.physical_address
my_ip.register_map.values_fea1_offset_1 = values_fea_buffer.physical_address
my_ip.register_map.values_fea2_offset_1 = values_fea_buffer.physical_address
my_ip.register_map.values_fea3_offset_1 = values_fea_buffer.physical_address
my_ip.register_map.values_fea4_offset_1 = values_fea_buffer.physical_address
my_ip.register_map.rowPtr_adj1_offset_1 = rowPtr_adj_buffer.physical_address
my_ip.register_map.rowPtr_adj2_offset_1 = rowPtr_adj_buffer.physical_address
my_ip.register_map.rowPtr_adj3_offset_1 = rowPtr_adj_buffer.physical_address
my_ip.register_map.rowPtr_adj4_offset_1 = rowPtr_adj_buffer.physical_address
my_ip.register_map.columnIndex_adj1_offset_1 = columnIndex_adj_buffer.physical_address
my_ip.register_map.columnIndex_adj2_offset_1 = columnIndex_adj_buffer.physical_address
my_ip.register_map.columnIndex_adj3_offset_1 = columnIndex_adj_buffer.physical_address
my_ip.register_map.columnIndex_adj4_offset_1 = columnIndex_adj_buffer.physical_address
my_ip.register_map.values_adj1_offset_1 = values_adj_buffer.physical_address
my_ip.register_map.values_adj2_offset_1 = values_adj_buffer.physical_address
my_ip.register_map.values_adj3_offset_1 = values_adj_buffer.physical_address
my_ip.register_map.values_adj4_offset_1 = values_adj_buffer.physical_address
my_ip.register_map.D1_offset_1 = D_buffer.physical_address
my_ip.register_map.D2_offset_1 = D_buffer.physical_address
my_ip.register_map.D3_offset_1 = D_buffer.physical_address
my_ip.register_map.D4_offset_1 = D_buffer.physical_address

if fix == 1:
    my_ip.register_map.scale_fea = float_to_fix(internal_scale, 0, 2, 31, 32)
else:
    my_ip.register_map.scale_fea = 0x3f800000

if fix == 1:
    my_ip.register_map.quantized_multiplier = float_to_fix(internal_quantization, -128, 127, ifrac, ibits)
    #my_ip.register_map.quantized_multiplier = internal_quantization
else:
    my_ip.register_map.quantized_multiplier = 0x42800000

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_weights = None
if custom == 1:
    transform = T.Compose([T.GCNNorm(), T.ToSparseTensor()])
else:
    transform = None

if dataset_sel == "Cora":
    ############################## Planetoid - Cora #######
    dataset = Planetoid(
        root="data/Planetoid", name="Cora", split="full", transform=transform
    )  # split = "public"/"full"
    labels = dataset._data.y.numpy()
    values, counts = np.unique(labels, return_counts=True)
    loss_weights = np.sum(counts) / counts
elif dataset_sel == "CiteSeer":
    ############################## Planetoid - CiteSeer ###
    dataset = Planetoid(
        root="data/Planetoid", name="CiteSeer", split="full", transform=transform
    )  # split = "public"/"full"
    labels = dataset._data.y.numpy()
    values, counts = np.unique(labels, return_counts=True)
    loss_weights = np.sum(counts) / counts
elif dataset_sel == "PubMed":
    ############################## Planetoid - PubMed #####
    dataset = Planetoid(root="data/Planetoid", name="PubMed", split="full", transform=transform
                        )
    labels = dataset._data.y.numpy()
    values, counts = np.unique(labels, return_counts=True)
    loss_weights = np.sum(counts) / counts
###########################################################################

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

criterion = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(loss_weights).to(device))

# =============== Train model =============================================#

if custom:
    model = GCN_PYNQ(hidden_channels, dataset.num_features, dataset.num_classes)
else:
    model = GCN_ref(dataset.num_features, hidden_channels, dataset.num_classes)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#scheduler = None
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=16, gamma=0.5)
model.to(device=device)

times = np.zeros((nrof_epochs, 3, 2))

train_loss_all, test_loss_all = [], []
accuracies = np.zeros((nrof_epochs, 1))
for epoch in range(nrof_epochs):
    train_loss = train(train_loader)
    test_loss, y_pred, y_true = eval(test_loader)
    train_loss_all.append(train_loss)
    test_loss_all.append(test_loss)
    acc_epoch = accuracy(np.argmax(y_pred, axis=1), y_true)
    accuracies[epoch] = acc_epoch
    print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {acc_epoch:.4f}")
    #print(f"Epoch: {epoch:03d}")
    if scheduler:
        scheduler.step()
#_, y_pred, y_true = eval(test_loader)
#acc_epoch = accuracy(np.argmax(y_pred, axis=1), y_true)
#print(f"Accuracy: {acc_epoch:.3f}")
#df = pd.concat([df, pd.DataFrame([[*combinations[i,:],acc_epoch]], columns=columns,),], ignore_index=True,)

#print(f"Percentage complete: {(i+1) / combinations.shape[0] * 100:.2f}%")
#if ((i % int(combinations.shape[0]/4)) == (int(combinations.shape[0]/4) - 1)):
#if ((i % int(combinations.shape[0]/2)) == (int(combinations.shape[0]/2) - 1)):
#if i == combinations.shape[0] - 1:
#        timestr = time.strftime("%Y%m%d-%H%M%S")
#        df.to_csv("accuracies/" + dataset_sel + "_sweep_" + timestr + ".csv", index=False)

# accuracy
if results_dir != None:
    results_file = results_dir + "/" + dataset_sel + "_" + design_config + "_" + str(reduced) + "_" + backward_config + ".csv"
    np.savetxt(results_file, accuracies, delimiter=",")

# times
if results_dir != None:
    times_file = results_dir + "/times/" + dataset_sel + "_" + design_config + "_" + str(reduced) + "_" + backward_config + ".csv"
    np.save(times_file, times)


values_adj_buffer.freebuffer()
values_fea_buffer.freebuffer()
B_buffer.freebuffer()
D_buffer.freebuffer()
rowPtr_fea_buffer.freebuffer()
columnIndex_fea_buffer.freebuffer()
rowPtr_adj_buffer.freebuffer()
columnIndex_adj_buffer.freebuffer()