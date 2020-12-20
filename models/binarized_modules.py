import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function
from torch.nn.parameter import Parameter

import numpy as np

def Binarize(tensor,quant_mode='det'):
    if quant_mode=='det':
        #return tensor.sign()
        return (tensor >= 0).type(tensor.type()) - (tensor < 0).type(tensor.type())
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)

class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss,self).__init__()
        self.margin=1.0

    def hinge_loss(self,input,target):
            #import pdb; pdb.set_trace()
            output=self.margin-input.mul(target)
            output[output.le(0)]=0
            return output.mean()

    def forward(self, input, target):
        return self.hinge_loss(input,target)

class SqrtHingeLossFunction(Function):
    def __init__(self):
        super(SqrtHingeLossFunction,self).__init__()
        self.margin=1.0

    def forward(self, input, target):
        output=self.margin-input.mul(target)
        output[output.le(0)]=0
        self.save_for_backward(input, target)
        loss=output.mul(output).sum(0).sum(1).div(target.numel())
        return loss

    def backward(self,grad_output):
       input, target = self.saved_tensors
       output=self.margin-input.mul(target)
       output[output.le(0)]=0
       import pdb; pdb.set_trace()
       grad_output.resize_as_(input).copy_(target).mul_(-2).mul_(output)
       grad_output.mul_(output.ne(0).float())
       grad_output.div_(input.numel())
       return grad_output,grad_output

def Quantize(tensor,quant_mode='det',  params=None, numBits=8):
    tensor.clamp_(-2**(numBits-1),2**(numBits-1))
    if quant_mode=='det':
        tensor=tensor.mul(2**(numBits-1)).round().div(2**(numBits-1))
    else:
        tensor=tensor.mul(2**(numBits-1)).round().add(torch.rand(tensor.size()).add(-0.5)).div(2**(numBits-1))
        quant_fixed(tensor, params)
    return tensor

class LSQ(Function):
    @staticmethod
    def forward(self, value, step_size, nbits):
        self.save_for_backward(value, step_size)
        self.other = nbits

        #set levels
        Qn = -2**(nbits-1)
        Qp = 2**(nbits-1) - 1

        v_bar = (value/step_size).round().clamp(Qn, Qp)
        v_hat = v_bar*step_size
        return v_hat

    @staticmethod
    def backward(self, grad_output):
        value, step_size = self.saved_tensors
        nbits = self.other

        #set levels
        Qn = -2**(nbits-1)
        Qp = 2**(nbits-1) - 1
        grad_scale = 1.0 / math.sqrt(value.numel() * Qp)

        lower = (value/step_size <= Qn).float()
        higher = (value/step_size >= Qp).float()
        middle = (1.0 - higher - lower)

        grad_step_size = lower*Qn + higher*Qp + middle*(-value/step_size + (value/step_size).round())

        return grad_output*middle, (grad_output*grad_step_size*grad_scale).sum().unsqueeze(dim=0), None

def grad_scale(x, scale):
    yOut = x
    yGrad = x*scale
    y = yOut.detach() - yGrad.detach() + yGrad
    return y

def round_pass(x):
    yOut = x.round()
    yGrad = x
    y = yOut.detach() - yGrad.detach() + yGrad
    return y

def quantizeLSQ(v, s, p):
    #set levels
    Qn = -2**(p-1)
    Qp = 2**(p-1) - 1
    if p==1 or p==-1: #-1 is ternary 
        Qn = -1
        Qp = 1
        gradScaleFactor = 1.0 / math.sqrt(v.numel())
    else:
        gradScaleFactor = 1.0 / math.sqrt(v.numel() * Qp)

    #quantize
    s = grad_scale(s, gradScaleFactor)
    vbar=round_pass((v/s).clamp(Qn, Qp))
    if p==1:
        vbar = Binarize(vbar)
    vhat = vbar*s
    return vhat

def custom_quantize(tensor, numBits=8):
    #tensor.clamp_(-2**(numBits),2**(numBits))
    temp = tensor.clone()
    maximum = temp.abs_().max()
    #print(tensor)
    delta =  torch.ceil((maximum/pow(2,numBits))*2)
    tensor.clamp_(-maximum,maximum-delta)			#clamp to required range
    #print(delta)
    #print(tensor)
    tensor = (tensor/delta).round() * delta

    return tensor


import torch.nn.functional as tnnf


class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):

        if input.size(1) != 784:
            input.data=Binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Binarize(self.weight.org)
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out

class _Conv2dQ(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super(_Conv2dQ, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                       padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.nbits = kwargs['nbits']
        self.quantize = kwargs['quantize']
        self.step_size = Parameter(torch.Tensor(1))

        #buffer is not updated for optim.step
        self.register_buffer('init_state', torch.zeros(1))

class BinarizeConv2d(_Conv2dQ):
    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        if input.size(1) != 3:
            input.data = Binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Binarize(self.weight.org)

        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        #LSQ QUANTIZATION
        if self.quantize:
            if self.init_state == 0:
                if self.nbits==1 or self.nbits==-1:
                    self.step_size.data.copy_(2*out.abs().mean() / math.sqrt(1.0))
                else:
                    self.step_size.data.copy_(2 * out.abs().mean() / math.sqrt(2**(self.nbits-1) - 1))
                self.init_state.fill_(1)

            #out = LSQ.apply(out, self.step_size, self.nbits)
            out = quantizeLSQ(out, self.step_size, self.nbits)

        return out



class BinarizeConv2d_with_ps(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)


    def forward(self, input):
        k = min(10,input.shape[1])
        if input.size(1) != 3:
            input.data = Binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Binarize(self.weight.org)
        weight = self.weight
        #out = nn.functional.conv2d(input, self.weight, None, self.stride,
        #                           self.padding, self.dilation, self.groups)
        output_ps = []

        print(input.shape)
        #calculating partial sums
        for i in range(k):
            input_temp = input[:,math.ceil(input.shape[1] / k) * i: min((math.ceil(input.shape[1] / k) * (i + 1)), input.shape[1]),:,:]
            weight_temp = weight[: , math.ceil(weight.shape[1] / k) * i: min((math.ceil(weight.shape[1] / k) * (i + 1) ), weight.shape[1]),:,:]
            #print(weight_temp.shape)
            #print(input_temp.shape)

            output = nn.functional.conv2d(input_temp,weight_temp,self.bias, self.stride, self.padding)
            output_ps.append(Quantize(output,numBits=5))

        output = output_ps[0]


        #combining partial sums

        for i in range(k-1):
            #print(output.shape)
            #print(output_ps[i].shape)

            #print("==========")
            #output_ps[i] =  Binarize(output_ps[i])
            output = output+output_ps[i+1]


        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            output += self.bias.view(1, -1, 1, 1).expand_as(output)

        return output
