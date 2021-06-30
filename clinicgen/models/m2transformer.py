#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import torch
import os
from torch import sigmoid
from torch.nn import Dropout, LayerNorm, Linear, Module
from torch.nn.functional import dropout, linear, relu, softmax
from torch.nn.init import normal_
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.transformer import _get_activation_fn, TransformerDecoderLayer, TransformerDecoder, TransformerEncoder
from torch.nn.parameter import Parameter
from clinicgen.models.transformer import _TransformerCaptioner
from clinicgen.models.gcnclassifier import GCNClassifier
# from clinicgen.models.mlclassifier import GCNClassifier
import random

import torch.nn as nn
import copy

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MeshedTransformerEncoder(TransformerEncoder):
    def __init__(self, encoder_layer, num_layers):
        super(MeshedTransformerEncoder, self).__init__(encoder_layer, num_layers)

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        outputs = []
        for i in range(self.num_layers):
            output = self.layers[i](output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            outputs.append(output)
        output = torch.stack(outputs, dim=1)
        return output

class MeshedTransformerDecoder(TransformerDecoder):
    def __init__(self, decoder_layer, num_layers):
        super(MeshedTransformerDecoder, self).__init__(decoder_layer, num_layers)

    def forward(self, tgt, memory, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None, require_attention_score=False):
        output = tgt

        # require attention score
        if require_attention_score:
            for mod in self.layers:
                output, atten_weight = mod(output, memory, tgt_mask=tgt_mask,
                             memory_mask=memory_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             memory_key_padding_mask=memory_key_padding_mask,
                             require_attention_score=require_attention_score)
            return output, atten_weight
        else:
            for mod in self.layers:
                output = mod(output, memory, tgt_mask=tgt_mask,
                             memory_mask=memory_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             memory_key_padding_mask=memory_key_padding_mask,
                             require_attention_score=require_attention_score)

            return output

# manually process attention weight
class TransformerAndAttention(MultiheadAttention):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super(TransformerAndAttention, self).__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim)
        pass

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):
        if not self._qkv_same_embed_dim:
            return self.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            return self.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)

    def multi_head_attention_forward(query, key, value, embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias, bias_k,
                                     bias_v, add_zero_attn, dropout_p, out_proj_weight, out_proj_bias, training=True,
                                     key_padding_mask=None, need_weights=True, attn_mask=None, use_separate_proj_weight=False,
                                     q_proj_weight=None, k_proj_weight=None, v_proj_weight=None, static_k=None, static_v=None):

        if not torch.jit.is_scripting():
            tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v,
                        out_proj_weight, out_proj_bias)
            if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(tens_ops):
                return handle_torch_function(
                    multi_head_attention_forward, tens_ops, query, key, value,
                    embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias,
                    bias_k, bias_v, add_zero_attn, dropout_p, out_proj_weight,
                    out_proj_bias, training=training, key_padding_mask=key_padding_mask,
                    need_weights=need_weights, attn_mask=attn_mask,
                    use_separate_proj_weight=use_separate_proj_weight,
                    q_proj_weight=q_proj_weight, k_proj_weight=k_proj_weight,
                    v_proj_weight=v_proj_weight, static_k=static_k, static_v=static_v)
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == embed_dim_to_check
        # allow MHA to have different sizes for the feature dimension
        assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        scaling = float(head_dim) ** -0.5

        if not use_separate_proj_weight:
            if torch.equal(query, key) and torch.equal(key, value):
                # self-attention
                q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

            elif torch.equal(key, value):
                # encoder-decoder attention
                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = 0
                _end = embed_dim
                _w = in_proj_weight[_start:_end, :]
                if _b is not None:
                    _b = _b[_start:_end]
                q = linear(query, _w, _b)

                if key is None:
                    assert value is None
                    k = None
                    v = None
                else:

                    # This is inline in_proj function with in_proj_weight and in_proj_bias
                    _b = in_proj_bias
                    _start = embed_dim
                    _end = None
                    _w = in_proj_weight[_start:, :]
                    if _b is not None:
                        _b = _b[_start:]
                    k, v = linear(key, _w, _b).chunk(2, dim=-1)

            else:
                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = 0
                _end = embed_dim
                _w = in_proj_weight[_start:_end, :]
                if _b is not None:
                    _b = _b[_start:_end]
                q = linear(query, _w, _b)

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = embed_dim * 2
                _w = in_proj_weight[_start:_end, :]
                if _b is not None:
                    _b = _b[_start:_end]
                k = linear(key, _w, _b)

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim * 2
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                v = linear(value, _w, _b)
        else:
            q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
            len1, len2 = q_proj_weight_non_opt.size()
            assert len1 == embed_dim and len2 == query.size(-1)

            k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
            len1, len2 = k_proj_weight_non_opt.size()
            assert len1 == embed_dim and len2 == key.size(-1)

            v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
            len1, len2 = v_proj_weight_non_opt.size()
            assert len1 == embed_dim and len2 == value.size(-1)

            if in_proj_bias is not None:
                q = linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
                k = linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:(embed_dim * 2)])
                v = linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):])
            else:
                q = linear(query, q_proj_weight_non_opt, in_proj_bias)
                k = linear(key, k_proj_weight_non_opt, in_proj_bias)
                v = linear(value, v_proj_weight_non_opt, in_proj_bias)
        q = q * scaling

        if attn_mask is not None:
            assert attn_mask.dtype == torch.float32 or attn_mask.dtype == torch.float64 or \
                   attn_mask.dtype == torch.float16 or attn_mask.dtype == torch.uint8 or attn_mask.dtype == torch.bool, \
                'Only float, byte, and bool types are supported for attn_mask, not {}'.format(attn_mask.dtype)
            if attn_mask.dtype == torch.uint8:
                warnings.warn(
                    "Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
                attn_mask = attn_mask.to(torch.bool)

            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 2D attn_mask is not correct.')
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 3D attn_mask is not correct.')
            else:
                raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
            # attn_mask's dim is 3 now.

        # convert ByteTensor key_padding_mask to bool
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            warnings.warn(
                "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            key_padding_mask = key_padding_mask.to(torch.bool)

        if bias_k is not None and bias_v is not None:
            if static_k is None and static_v is None:
                k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
                v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
                if attn_mask is not None:
                    attn_mask = pad(attn_mask, (0, 1))
                if key_padding_mask is not None:
                    key_padding_mask = pad(key_padding_mask, (0, 1))
            else:
                assert static_k is None, "bias cannot be added to static key."
                assert static_v is None, "bias cannot be added to static value."
        else:
            assert bias_k is None
            assert bias_v is None

        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        if static_k is not None:
            assert static_k.size(0) == bsz * num_heads
            assert static_k.size(2) == head_dim
            k = static_k

        if static_v is not None:
            assert static_v.size(0) == bsz * num_heads
            assert static_v.size(2) == head_dim
            v = static_v

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if add_zero_attn:
            src_len += 1
            k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
            v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float('-inf'))
            else:
                attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

        attn_output_weights_out = sigmoid(attn_output_weights)

        attn_output_weights = softmax(
            attn_output_weights, dim=-1)
        attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

        if need_weights:
            # average attention weights over heads
            attn_output_weights_out = attn_output_weights_out.view(bsz, num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights_out.sum(dim=1) / num_heads
        else:
            return attn_output, None

class MeshedTransformerMaxDecoderLayer(TransformerDecoderLayer):
    def __init__(self, d_model, nhead, nlayer_enc, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(MeshedTransformerMaxDecoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, activation)
        self.self_attn = TransformerAndAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = TransformerAndAttention(d_model, nhead, dropout=dropout)
        for i in range(nlayer_enc):
            setattr(self, 'linear_alpha{0}'.format(i), Linear(d_model * 2, d_model))

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None, require_attention_score=False):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        mesh = None
        for i in range(memory.shape[0]):
            if len(memory.shape) == 5:
                tgt_stack = []
                atten_weights_stack = []
                for j in range(memory.shape[1]):
                    tgt2, attn_weights = self.multihead_attn(tgt, memory[i, j], memory[i, j], attn_mask=memory_mask,
                                               key_padding_mask=memory_key_padding_mask)
                    tgt_stack.append(tgt2)
                    atten_weights_stack.append(attn_weights)
                tgt_stack = torch.stack(tgt_stack, dim=0)
                tgt2, _ = torch.max(tgt_stack, dim=0)
                atten_weights_stack = torch.stack(atten_weights_stack, dim=0)
                atten_weights, _ = torch.max(atten_weights_stack, dim=0) #is that right ?
            else:
                tgt2, attn_weights = self.multihead_attn(tgt, memory[i], memory[i], attn_mask=memory_mask,
                                           key_padding_mask=memory_key_padding_mask)
            mtgt = tgt + self.dropout2(tgt2)
            mtgt = self.norm2(mtgt)
            linear_alpha = getattr(self, 'linear_alpha{0}'.format(i))
            alpha = sigmoid(linear_alpha(torch.cat([tgt, mtgt], dim=2)))
            if mesh is None:
                mesh = mtgt * alpha
            else:
                mesh += mtgt * alpha
        mesh *= memory.shape[0] ** -0.5
        tgt = mesh

        if hasattr(self, "activation"):
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        else:  # for backward compatibility
            tgt2 = self.linear2(self.dropout(relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        # attention score for words
        if require_attention_score:
            atten_weights = atten_weights.permute(1, 0, 2)
            atten_score = torch.mean(atten_weights, 2)
            return tgt, atten_score
        else:
            return tgt


class MultiheadAttentionWithMem(MultiheadAttention):
    def __init__(self, embed_dim, num_heads, num_memory, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None):
        super(MultiheadAttentionWithMem, self).__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn,
                                                        kdim, vdim)
        self.num_memory = num_memory
        self.mem_k = Parameter(torch.Tensor(num_memory, 1, embed_dim))
        self.mem_v = Parameter(torch.Tensor(num_memory, 1, embed_dim))
        self._reset_mem_parameters()

    def _reset_mem_parameters(self):
        normal_(self.mem_k, 0.0, 1.0 / (self.embed_dim // self.num_heads))
        normal_(self.mem_v, 0.0, 1.0 / self.num_memory)

    @staticmethod
    def multi_head_attention_forward(query, key, value, embed_dim_to_check, num_heads, num_memory, in_proj_weight,
                                     in_proj_bias, bias_k, bias_v, mem_k, mem_v, add_zero_attn, dropout_p,
                                     out_proj_weight, out_proj_bias, training=True, key_padding_mask=None,
                                     need_weights=True, attn_mask=None, use_separate_proj_weight=False,
                                     q_proj_weight=None, k_proj_weight=None, v_proj_weight=None, static_k=None,
                                     static_v=None):
        qkv_same = torch.equal(query, key) and torch.equal(key, value)
        kv_same = torch.equal(key, value)

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == embed_dim_to_check
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        scaling = float(head_dim) ** -0.5

        if use_separate_proj_weight is not True:
            if qkv_same:
                # self-attention
                q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

            elif kv_same:
                # encoder-decoder attention
                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = 0
                _end = embed_dim
                _w = in_proj_weight[_start:_end, :]
                if _b is not None:
                    _b = _b[_start:_end]
                q = linear(query, _w, _b)

                if key is None:
                    assert value is None
                    k = None
                    v = None
                else:

                    # This is inline in_proj function with in_proj_weight and in_proj_bias
                    _b = in_proj_bias
                    _start = embed_dim
                    _end = None
                    _w = in_proj_weight[_start:, :]
                    if _b is not None:
                        _b = _b[_start:]
                    k, v = linear(key, _w, _b).chunk(2, dim=-1)

            else:
                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = 0
                _end = embed_dim
                _w = in_proj_weight[_start:_end, :]
                if _b is not None:
                    _b = _b[_start:_end]
                q = linear(query, _w, _b)

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = embed_dim * 2
                _w = in_proj_weight[_start:_end, :]
                if _b is not None:
                    _b = _b[_start:_end]
                k = linear(key, _w, _b)

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim * 2
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                v = linear(value, _w, _b)
        else:
            q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
            len1, len2 = q_proj_weight_non_opt.size()
            assert len1 == embed_dim and len2 == query.size(-1)

            k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
            len1, len2 = k_proj_weight_non_opt.size()
            assert len1 == embed_dim and len2 == key.size(-1)

            v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
            len1, len2 = v_proj_weight_non_opt.size()
            assert len1 == embed_dim and len2 == value.size(-1)

            if in_proj_bias is not None:
                q = linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
                k = linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:(embed_dim * 2)])
                v = linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):])
            else:
                q = linear(query, q_proj_weight_non_opt, in_proj_bias)
                k = linear(key, k_proj_weight_non_opt, in_proj_bias)
                v = linear(value, v_proj_weight_non_opt, in_proj_bias)
        q = q * scaling

        if bias_k is not None and bias_v is not None:
            if static_k is None and static_v is None:
                k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
                v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
                if attn_mask is not None:
                    attn_mask = torch.cat([attn_mask,
                                           torch.zeros((attn_mask.size(0), 1),
                                                       dtype=attn_mask.dtype,
                                                       device=attn_mask.device)], dim=1)
                if key_padding_mask is not None:
                    key_padding_mask = torch.cat(
                        [key_padding_mask, torch.zeros((key_padding_mask.size(0), 1),
                                                       dtype=key_padding_mask.dtype,
                                                       device=key_padding_mask.device)], dim=1)
            else:
                assert static_k is None, "bias cannot be added to static key."
                assert static_v is None, "bias cannot be added to static value."
        else:
            assert bias_k is None
            assert bias_v is None

        # Additional memory process
        scaled_mem_k = ((embed_dim // num_heads) ** 0.5) * mem_k.expand(num_memory, k.shape[1], k.shape[2])
        scaled_mem_v = (num_memory ** 0.5) * mem_v.expand(num_memory, v.shape[1], v.shape[2])
        k = torch.cat([k, scaled_mem_k], dim=0)
        v = torch.cat([v, scaled_mem_v], dim=0)

        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        if static_k is not None:
            assert static_k.size(0) == bsz * num_heads
            assert static_k.size(2) == head_dim
            k = static_k

        if static_v is not None:
            assert static_v.size(0) == bsz * num_heads
            assert static_v.size(2) == head_dim
            v = static_v

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if add_zero_attn:
            src_len += 1
            k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
            v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, torch.zeros((attn_mask.size(0), 1),
                                                              dtype=attn_mask.dtype,
                                                              device=attn_mask.device)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros((key_padding_mask.size(0), 1),
                                                   dtype=key_padding_mask.dtype,
                                                   device=key_padding_mask.device)], dim=1)

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

        attn_output_weights = softmax(
            attn_output_weights, dim=-1)
        attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights.sum(dim=1) / num_heads
        else:
            return attn_output, None

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        if hasattr(self, '_qkv_same_embed_dim') and self._qkv_same_embed_dim is False:
            return self.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads, self.num_memory,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.mem_k, self.mem_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            if not hasattr(self, '_qkv_same_embed_dim'):
                warnings.warn('A new version of MultiheadAttention module has been implemented. \
                    Please re-train your model with the new module',
                              UserWarning)

            return self.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads, self.num_memory,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.mem_k, self.mem_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)


class TransformerEncoderLayerWithMem(Module):
    def __init__(self, d_model, nhead, nmem, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayerWithMem, self).__init__()
        self.self_attn = MultiheadAttentionWithMem(d_model, nhead, nmem, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        if hasattr(self, "activation"):
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        else:  # for backward compatibility
            src2 = self.linear2(self.dropout(relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class M2Transformer(_TransformerCaptioner):
    def __init__(self, embeddings, feat_dim=512, max_word=32, multi_image=1, layer_norm=False, num_memory=40,
                 num_enc_layers=6, num_dec_layers=6, teacher_forcing=False, image_model=None, image_pretrained=None,
                 finetune_image=False, image_finetune_epoch=None, rl_opts=None, word_idxs=None, device='gpu',
                 verbose=False,cls_pretrained=None,kg_dir=None,feed_mode='both'):#, feed_mode = 'both'):
        super(M2Transformer, self).__init__(embeddings, feat_dim, max_word, multi_image, False, layer_norm,
                                            teacher_forcing, image_model, image_pretrained, finetune_image,
                                            image_finetune_epoch, rl_opts, word_idxs, device, verbose)
        # # self.feed_mode = feed_mode
        # # OVERRIDE Some Layers
        # self.dropout = clones(Dropout(0.1), 2)
        # self.layer_norm = clones(LayerNorm(feat_dim) , 2)if layer_norm else None
        # self.image_proj_l = clones(Linear(self.image_dim, feat_dim), 2)

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayerWithMem(feat_dim, nhead=8, nmem=num_memory)
        self.encoder = MeshedTransformerEncoder(encoder_layer, num_layers=num_enc_layers)
        # Transformer Decoder
        decoder_layer = MeshedTransformerMaxDecoderLayer(feat_dim, nhead=8, nlayer_enc=num_enc_layers)
        self.decoder = MeshedTransformerDecoder(decoder_layer, num_layers=num_dec_layers)
        self.gcncls = GCNClassifier(kg_dir=kg_dir) #edit
        if cls_pretrained is not None and os.path.exists(cls_pretrained):
            pretrained = torch.load(cls_pretrained)
            pretrained_state_dict = pretrained['model_state_dict']
            state_dict = self.gcncls.state_dict()
            state_dict.update({k: v for k, v in pretrained_state_dict.items() if k in state_dict and 'fc' not in k})
            self.gcncls.load_state_dict(state_dict)
            print('load pretrained classifier model from {} successfully!'.format(cls_pretrained))
        self.feed_mode=feed_mode
    # def feed_mode_controller(self, att_feats, node_feats):
    #     assert self.feed_mode != None
    #
    #     if self.feed_mode == 'both':
    #         if self.encoder_mode in ['dualwayencoder', 'xdualwayencoder']:
    #             input_feats = [att_feats, node_feats]
    #         else:
    #             input_feats = torch.cat((att_feats, node_feats), dim=1)  # torch.Size([16, 70, 2048])
    #     # feed only CNN features
    #     elif self.feed_mode == 'cnn_only':
    #         input_feats = att_feats
    #     # feed only graph embedded features
    #     elif self.feed_mode == 'gcn_only':
    #         input_feats = node_feats
    #     return input_feats


    def decode_beam(self, encoded_data, beam_size, allow_stop=True, recover_words=None, diversity_rate=0.0,require_attention_score=None):
        return super(M2Transformer, self).decode_beam(encoded_data, beam_size, allow_stop, recover_words,
                                                      diversity_rate,require_attention_score)

    def deflatten_image(self, x):
        if self.multi_image > 1:
            x = x.view(int(x.shape[0] / self.multi_image), self.multi_image, self.encoder.num_layers, x.shape[2],
                       x.shape[3])
        return x

    def encode_image(self, x,train):
        # CNN+Transformer features
        x = self.flatten_image(x)
        v, _ = self.image_features_with_mask(x, self.image_feats,train=train)
        # Merge multiple images
        v = self.deflatten_image(v)
        return v

    def image_features_with_mask(self, x, model,train=True):
        mask = (x.detach().sum(dim=(1, 2, 3)) != 0).type(torch.float).unsqueeze(dim=-1).unsqueeze(dim=-1)
        if self.multi_image > 1:
            nz = mask.squeeze().nonzero().squeeze()
            x_nz = x[nz]
            if len(nz.shape) == 0:
                x_nz = x_nz.unsqueeze(dim=0)
        else:
            x_nz, nz = x, None

        batch_size = int(x.shape[0]/self.multi_image)
        remain_img_num = int(x_nz.shape[0]/batch_size)

        cnn_feats_comb = []
        node_feats_comb = []
        node_states_comb = []
        global_states_comb = []
        img_list = [i for i in range(remain_img_num)]
        if train == True:  #edit: if train, shuffle
            random.shuffle(img_list)
        for i in img_list:
            #GCN
            cnn_feats,node_feats,node_states,global_states = self.gcncls(x_nz[i*batch_size:(i+1)*batch_size])
            cnn_feats_comb.append(cnn_feats)
            node_feats_comb.append(node_feats)
            node_states_comb.append(node_states)
            global_states_comb.append(global_states.unsqueeze(1))
        cnn_feats_comb=torch.cat(cnn_feats_comb, dim=0) # Key Feats
        node_feats_comb = torch.cat(node_feats_comb, dim=0)
        node_states_comb= torch.cat(node_states_comb, dim=0) # Key Feats
        global_states_comb= torch.cat(global_states_comb, dim=0)

        all_feats = torch.cat([cnn_feats_comb,node_states_comb],dim = 1)

        if self.feed_mode=='both':
            input_feats=all_feats
        elif self.feed_mode=='cnn_only':
            input_feats = cnn_feats_comb
        elif self.feed_mode=='gcn_only':
            input_feats = node_states_comb

        # use cnn and gcn out put as input of attention
        input_feats = relu(self.image_proj_l(input_feats))
        input_feats = self.dropout(input_feats)
        if self.layer_norm is not None:
            input_feats= self.layer_norm(input_feats)
        input_feats = input_feats.permute(1, 0, 2)

        # node_states_comb = relu(self.image_proj_l(node_states_comb))
        # node_states_comb = self.dropout(node_states_comb)
        # if self.layer_norm is not None:
        #     node_states_comb = self.layer_norm(node_states_comb)
        # node_states_comb = node_states_comb.permute(1, 0, 2)

        # Transformer encoder
        # x_nz = self.encoder(node_feats_comb, node_states_comb)  # make cnn as q, gcn as k,v
        x_nz = self.encoder(input_feats)
        x_nz = x_nz.permute(1, 2, 0, 3)
        xms = []
        if self.multi_image > 1:
            for i in range(self.encoder.num_layers):
                xm = x.new_zeros(x.shape[0], x_nz.shape[2], x_nz.shape[3])
                xm[nz] += x_nz[i]
                xms.append(xm)
        else:
            for i in range(self.encoder.num_layers):
                xms.append(x_nz[i])
        x = torch.stack(xms, dim=1)
        return x, mask

    def proc_word_sequence(self, wt, v, mask=None, require_attention_score=False):
        e = self.embeddings(wt)
        # Positional encoding + Transformer decoder
        e = e.permute(1, 0, 2)
        e *= self.word_weight
        e = self.word_pe(e)
        if self.multi_image > 1:
            v2 = v.permute(2, 1, 3, 0, 4)
        else:
            v2 = v.permute(1, 2, 0, 3)
        return self.decoder(e, v2, tgt_mask=mask, require_attention_score=require_attention_score)
