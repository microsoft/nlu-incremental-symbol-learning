# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn.functional as F

def miso_multi_head_attention_forward(query,                           # type: Tensor
                                      key,                             # type: Tensor
                                      value,                           # type: Tensor
                                      embed_dim_to_check,              # type: int
                                      num_heads,                       # type: int
                                      in_proj_weight,                  # type: Tensor
                                      in_proj_bias,                    # type: Tensor
                                      bias_k,                          # type: Optional[Tensor]
                                      bias_v,                          # type: Optional[Tensor]
                                      add_zero_attn,                   # type: bool
                                      dropout_p,                       # type: float
                                      out_proj_weight,                 # type: Tensor
                                      out_proj_bias,                   # type: Tensor
                                      training=True,                   # type: bool
                                      key_padding_mask=None,           # type: Optional[Tensor]
                                      need_weights=True,               # type: bool
                                      attn_mask=None,                  # type: Optional[Tensor]
                                      use_separate_proj_weight=False,  # type: bool
                                      q_proj_weight=None,              # type: Optional[Tensor]
                                      k_proj_weight=None,              # type: Optional[Tensor]
                                      v_proj_weight=None,              # type: Optional[Tensor]
                                      static_k=None,                   # type: Optional[Tensor]
                                      static_v=None,                   # type: Optional[Tensor]
                                      # my additions!
                                      p_proj_weight_1=None,            # type: Optional[Tensor]
                                      p_proj_weight_2=None,            # type: Optional[Tensor]
                                      p_value=None                     # type: Optional[Tensor]
                                      ):
    # type: (...) -> Tuple[Tensor, Optional[Tensor]]
    r"""Modified from https://github.com/pytorch/pytorch/blob/3a63a939d4d2549fc970725e7d16d9c44a0314a8/torch/nn/functional.py#L3849
    to support graph-positional encoding à la https://arxiv.org/pdf/1911.03561.pdf
    """
    #if not torch.jit.is_scripting():
    #    tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v,
    #                out_proj_weight, out_proj_bias)
    #    if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(tens_ops):
    #        return handle_torch_function(
    #            multi_head_attention_forward, tens_ops, query, key, value,
    #            embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias,
    #            bias_k, bias_v, add_zero_attn, dropout_p, out_proj_weight,
    #            out_proj_bias, training=training, key_padding_mask=key_padding_mask,
    #            need_weights=need_weights, attn_mask=attn_mask,
    #            use_separate_proj_weight=use_separate_proj_weight,
    #            q_proj_weight=q_proj_weight, k_proj_weight=k_proj_weight,
    #            v_proj_weight=v_proj_weight, static_k=static_k, static_v=static_v)
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
            q, k, v = F.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        elif torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

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
                k, v = F.linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = F.linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = F.linear(value, _w, _b)
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
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:(embed_dim * 2)])
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):])
        else:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if attn_mask is not None:
        assert attn_mask.dtype == torch.float32 or attn_mask.dtype == torch.float64 or \
            attn_mask.dtype == torch.float16 or attn_mask.dtype == torch.uint8 or attn_mask.dtype == torch.bool, \
            'Only float, byte, and bool types are supported for attn_mask, not {}'.format(attn_mask.dtype)
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
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
        warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
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

    # TODO: graph positional encoding here 
    p_qk = F.linear(p_value, p_proj_weight_1)

    # first step: add relative positional encoding before bmm
    prev_shape = k.shape
    k = k.reshape((bsz, num_heads, -1, head_dim)) 
    k = k + p_qk
    k = k.reshape(prev_shape) 


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

    attn_output_weights = F.softmax(
        attn_output_weights, dim=-1)
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)

    # TODO: add more graph positional encoding here
    p_kv = F.linear(p_value, p_proj_weight_2)

    # second step: add second projection to value
    prev_shape = v.shape
    v = v.reshape((bsz, num_heads, -1, head_dim))
    v = v + p_kv 
    v = v.reshape(prev_shape)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None

class MisoMultiheadAttention(torch.nn.MultiheadAttention):
    r"""Modified torch.nn.MultiheadAttention so that we can have
    graph positional encodings. Identical except for call to miso_multi_head_attention_forward
    rather than torch.functional.multi_head_attention_forward"""

    __annotations__ = {
        'bias_k': torch._jit_internal.Optional[torch.Tensor],
        'bias_v': torch._jit_internal.Optional[torch.Tensor],
    }

    def __init__(self, embed_dim, num_heads, num_ops, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super(MisoMultiheadAttention, self).__init__(embed_dim, num_heads, 
                                                    dropout=dropout, bias=bias, 
                                                    add_bias_kv=add_bias_kv, 
                                                    add_zero_attn=add_zero_attn, 
                                                    kdim=kdim, vdim=vdim) 
        
        proj_dim = int(embed_dim/num_heads)
        self.p_proj_weight_1 = torch.nn.Parameter(torch.Tensor(proj_dim, num_ops))
        self.p_proj_weight_2 = torch.nn.Parameter(torch.Tensor(proj_dim, num_ops))

    def forward(self, query, key, value, op_vec, key_padding_mask=None,
                need_weights=True, attn_mask=None):

        if not self._qkv_same_embed_dim:
            return miso_multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                p_value=op_vec,
                p_proj_weight_1=self.p_proj_weight_1,
                p_proj_weight_2=self.p_proj_weight_2
                )
        else:
            return miso_multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask,
                p_value=op_vec,
                p_proj_weight_1=self.p_proj_weight_1,
                p_proj_weight_2=self.p_proj_weight_2
                )

