"""fused Triton kernels for gather-style block-sparse attention
(per-token top-k kv blocks, single shared index head).

Adapted from flash-linear-attention's NSA kernels (fla/ops/nsa/parallel.py), with
three modifications for this model:
  1. Small GQA group support: HQ/Hkv = 2 here, but tl.dot needs M >= 16, so the
     head-group tile is padded to GP=16 with masked loads/stores (the kernel is
     memory-bound on K/V block loads, the padded tensor-core lanes are ~free).
  2. Multi index-head support ([B, HI, T, S] int32, -1 = pad). Each main-attn head
     hm uses index head hm // (HQ // HI); equivalently kv head i_h uses index head
     (i_h * G) // (HQ // HI). HI == 1 reduces to the shared-index-head case.
     Requires num_kv_heads % HI == 0 so a kv-head tile maps to one index head.
  3. No varlen / block_counts support (single contiguous causal sequence; the
     Python wrapper falls back to the eager gather path otherwise).

Semantics match Qwen3Indexer.gather_sparse_attention's eager path exactly:
per-token topk blocks, token-level causality inside blocks, duplicated block ids
collapsed (deduped to -1 before the kernel), rows with an empty candidate set
output zeros.
"""

import torch
import triton
import triton.language as tl

SPARSE_BLOCK_SIZE_SENTINEL = 128


@triton.jit(do_not_specialize=['T'])
def _gather_attn_fwd_kernel(
    q, k, v, o, lse, scale, block_indices, seg,
    T,
    H: tl.constexpr, HQ: tl.constexpr, G: tl.constexpr, GP: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr, S: tl.constexpr,
    BS: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    LW: tl.constexpr, NLB: tl.constexpr,
    HI: tl.constexpr, IDXG: tl.constexpr,
):
    i_t, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T
    seg_start = tl.load(seg + bos + i_t).to(tl.int32)

    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    i_idx = (i_h * G) // IDXG
    block_indices += ((i_b * HI + i_idx) * T + i_t) * S

    o_g = tl.arange(0, GP)
    o_k = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    m_g = o_g < G
    m_k = o_k < K
    m_v = (i_v * BV + o_v) < V

    p_q = q + (bos + i_t) * HQ * K + (i_h * G + o_g)[:, None] * K + o_k[None, :]
    b_q = tl.load(p_q, mask=m_g[:, None] & m_k[None, :], other=0.)
    b_q = (b_q * scale).to(b_q.dtype)

    b_o = tl.zeros([GP, BV], dtype=tl.float32)
    b_m = tl.full([GP], float('-inf'), dtype=tl.float32)
    b_acc = tl.zeros([GP], dtype=tl.float32)
    ls = tl.maximum(i_t - LW + 1, seg_start)
    for i in range(S):
        i_s = tl.load(block_indices + i).to(tl.int32) * BS
        if i_s <= i_t and i_s >= 0:
            p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (0, i_s), (BK, BS), (0, 1))
            p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_s = tl.dot(b_q, b_k)
            key = i_s + tl.arange(0, BS)
            b_s = tl.where(((i_t >= key) & (key < ls) & (key >= seg_start))[None, :], b_s, float('-inf'))
            b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
            b_r = tl.exp(b_mp - b_m)
            b_p = tl.exp(b_s - b_m[:, None])
            b_acc = b_acc * b_r + tl.sum(b_p, 1)
            b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)
    q_blk = i_t // BS
    for lb in range(NLB):
        i_s = (q_blk - lb) * BS
        if i_s >= 0:
            p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (0, i_s), (BK, BS), (0, 1))
            p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_s = tl.dot(b_q, b_k)
            key = i_s + tl.arange(0, BS)
            b_s = tl.where(((i_t >= key) & (key >= ls))[None, :], b_s, float('-inf'))
            b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
            b_r = tl.exp(b_mp - b_m)
            b_p = tl.exp(b_s - b_m[:, None])
            b_acc = b_acc * b_r + tl.sum(b_p, 1)
            b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)

    b_safe = tl.where(b_acc > 0, b_acc, 1.0)
    b_o = b_o / b_safe[:, None]
    b_lse = b_m + tl.log(b_safe)

    p_o = o + (bos + i_t) * HQ * V + (i_h * G + o_g)[:, None] * V + (i_v * BV + o_v)[None, :]
    tl.store(p_o, b_o.to(q.dtype.element_ty), mask=m_g[:, None] & m_v[None, :])
    if i_v == 0:
        p_lse = lse + (bos + i_t) * HQ + i_h * G + o_g
        tl.store(p_lse, b_lse, mask=m_g)


@triton.jit(do_not_specialize=['T'])
def _gather_attn_bwd_dq_kernel(
    q, k, v, lse, delta, do, dq, scale, block_indices, seg,
    T,
    H: tl.constexpr, HQ: tl.constexpr, G: tl.constexpr, GP: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr, S: tl.constexpr,
    BS: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    LW: tl.constexpr, NLB: tl.constexpr,
    HI: tl.constexpr, IDXG: tl.constexpr,
):
    i_t, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T
    seg_start = tl.load(seg + bos + i_t).to(tl.int32)

    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    i_idx = (i_h * G) // IDXG
    block_indices += ((i_b * HI + i_idx) * T + i_t) * S

    o_g = tl.arange(0, GP)
    o_k = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    m_g = o_g < G
    m_k = o_k < K
    m_v = (i_v * BV + o_v) < V

    p_q = q + (bos + i_t) * HQ * K + (i_h * G + o_g)[:, None] * K + o_k[None, :]
    b_q = tl.load(p_q, mask=m_g[:, None] & m_k[None, :], other=0.)
    b_q = (b_q * scale).to(b_q.dtype)

    p_do = do + (bos + i_t) * HQ * V + (i_h * G + o_g)[:, None] * V + (i_v * BV + o_v)[None, :]
    b_do = tl.load(p_do, mask=m_g[:, None] & m_v[None, :], other=0.)
    p_lse = lse + (bos + i_t) * HQ + i_h * G + o_g
    b_lse = tl.load(p_lse, mask=m_g, other=float('inf'))
    p_delta = delta + (bos + i_t) * HQ + i_h * G + o_g
    b_delta = tl.load(p_delta, mask=m_g, other=0.)

    b_dq = tl.zeros([GP, BK], dtype=tl.float32)
    ls = tl.maximum(i_t - LW + 1, seg_start)
    for i in range(S):
        i_s = tl.load(block_indices + i).to(tl.int32) * BS
        if i_s <= i_t and i_s >= 0:
            p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (0, i_s), (BK, BS), (0, 1))
            p_v = tl.make_block_ptr(v, (V, T), (1, H * V), (i_v * BV, i_s), (BV, BS), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_s = tl.dot(b_q, b_k)
            key = i_s + tl.arange(0, BS)
            b_p = tl.exp(b_s - b_lse[:, None])
            b_p = tl.where(((i_t >= key) & (key < ls) & (key >= seg_start))[None, :], b_p, 0.)
            b_dp = tl.dot(b_do, b_v)
            b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
            b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))
    q_blk = i_t // BS
    for lb in range(NLB):
        i_s = (q_blk - lb) * BS
        if i_s >= 0:
            p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (0, i_s), (BK, BS), (0, 1))
            p_v = tl.make_block_ptr(v, (V, T), (1, H * V), (i_v * BV, i_s), (BV, BS), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_s = tl.dot(b_q, b_k)
            key = i_s + tl.arange(0, BS)
            b_p = tl.exp(b_s - b_lse[:, None])
            b_p = tl.where(((i_t >= key) & (key >= ls))[None, :], b_p, 0.)
            b_dp = tl.dot(b_do, b_v)
            b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
            b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))
    b_dq *= scale

    p_dq = dq + (bos + i_t) * HQ * K + (i_h * G + o_g)[:, None] * K + o_k[None, :]
    tl.store(p_dq, b_dq.to(q.dtype.element_ty), mask=m_g[:, None] & m_k[None, :])


@triton.jit(do_not_specialize=['T'])
def _gather_attn_bwd_dkv_kernel(
    q, k, v, lse, delta, do, dk, dv, block_mask, seg, scale,
    T,
    H: tl.constexpr, HQ: tl.constexpr, G: tl.constexpr, GP: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr, M: tl.constexpr,
    BS: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    LW: tl.constexpr,
    HI: tl.constexpr, IDXG: tl.constexpr,
):
    i_v, i_s, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T
    i_idx = (i_h * G) // IDXG

    p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_s * BS, 0), (BS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_s * BS, i_v * BV), (BS, BV), (1, 0))
    p_dk = tl.make_block_ptr(dk + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_s * BS, 0), (BS, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_s * BS, i_v * BV), (BS, BV), (1, 0))

    o_g = tl.arange(0, GP)
    o_k = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    m_g = o_g < G
    m_k = o_k < K
    m_v = (i_v * BV + o_v) < V

    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dk = tl.zeros([BS, BK], dtype=tl.float32)
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dv = tl.zeros([BS, BV], dtype=tl.float32)

    for i in range(i_s * BS, T):
        b_flag = tl.load(block_mask + ((i_b * HI + i_idx) * T + i) * M + i_s)
        _proc = (b_flag != 0) | (i < (i_s + 1) * BS + LW)
        if _proc:
            p_q = q + (bos + i) * HQ * K + (i_h * G + o_g)[:, None] * K + o_k[None, :]
            b_q = tl.load(p_q, mask=m_g[:, None] & m_k[None, :], other=0.)
            b_q = (b_q * scale).to(b_q.dtype)

            p_do = do + (bos + i) * HQ * V + (i_h * G + o_g)[:, None] * V + (i_v * BV + o_v)[None, :]
            b_do = tl.load(p_do, mask=m_g[:, None] & m_v[None, :], other=0.)
            p_lse = lse + (bos + i) * HQ + i_h * G + o_g
            b_lse = tl.load(p_lse, mask=m_g, other=float('inf'))
            p_delta = delta + (bos + i) * HQ + i_h * G + o_g
            b_delta = tl.load(p_delta, mask=m_g, other=0.)

            b_s = tl.dot(b_k, tl.trans(b_q))
            b_p = tl.exp(b_s - b_lse[None, :])
            seg_i = tl.load(seg + bos + i).to(tl.int32)
            key = i_s * BS + tl.arange(0, BS)            
            causal = i >= key                             
            in_seg = key >= seg_i                         
            is_local = key > (i - LW)                     
            keep = causal & in_seg & (is_local | (b_flag != 0))
            b_p = tl.where(keep[:, None], b_p, 0.)
            b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
            b_dp = tl.dot(b_v, tl.trans(b_do))
            b_ds = b_p * (b_dp.to(tl.float32) - b_delta[None, :])
            b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)

    tl.store(p_dk, b_dk.to(q.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(q.dtype.element_ty), boundary_check=(0, 1))


def _kernel_meta(q, k, v, sel, block_size):
    B, T, HQ, K = q.shape
    H = k.shape[2]
    V = v.shape[3]
    S = sel.shape[-1]
    G = HQ // H
    GP = max(16, triton.next_power_of_2(G))
    BK = triton.next_power_of_2(K)
    BV = min(128, triton.next_power_of_2(V))
    if BV < V:
        raise NotImplementedError(f"head_dim V={V} > 128 not supported")
    if BK > 256:
        raise NotImplementedError(f"head_dim K={K} > 256 not supported")
    return B, T, HQ, H, K, V, S, G, GP, block_size, BK, BV


def _local_meta(block_size, local_window):
    """local window length in tokens + how many physical blocks can overlap it.
    The window [pos - LW + 1, pos] is not block-aligned, so it spans at most
    ceil(LW / BS) + 1 blocks; with LW a multiple of BS that is LW // BS + 1."""
    lw = block_size if local_window is None else int(local_window)
    lw = max(1, lw)
    nlb = (lw + block_size - 1) // block_size + 1
    return lw, nlb


def _gather_attn_fwd(q, k, v, sel, seg, block_size, scale, local_window=None):
    B, T, HQ, H, K, V, S, G, GP, BS, BK, BV = _kernel_meta(q, k, v, sel, block_size)
    HI = sel.shape[1]                                                        
    IDXG = HQ // HI                                             
    LW, NLB = _local_meta(BS, local_window)
    o = torch.empty(B, T, HQ, V, dtype=q.dtype, device=q.device)
    lse = torch.empty(B, T, HQ, dtype=torch.float32, device=q.device)
    _gather_attn_fwd_kernel[(T, 1, B * H)](
        q, k, v, o, lse, scale, sel, seg,
        T=T, H=H, HQ=HQ, G=G, GP=GP, K=K, V=V, S=S, BS=BS, BK=BK, BV=BV,
        LW=LW, NLB=NLB, HI=HI, IDXG=IDXG,
        num_warps=4,
    )
    return o, lse


def _gather_attn_bwd(q, k, v, o, lse, do, sel, seg, block_size, scale, local_window=None):
    B, T, HQ, H, K, V, S, G, GP, BS, BK, BV = _kernel_meta(q, k, v, sel, block_size)
    HI = sel.shape[1]
    IDXG = HQ // HI
    LW, NLB = _local_meta(BS, local_window)
    delta = (o.float() * do.float()).sum(-1)              

    dq = torch.empty_like(q)
    _gather_attn_bwd_dq_kernel[(T, 1, B * H)](
        q, k, v, lse, delta, do, dq, scale, sel, seg,
        T=T, H=H, HQ=HQ, G=G, GP=GP, K=K, V=V, S=S, BS=BS, BK=BK, BV=BV,
        LW=LW, NLB=NLB, HI=HI, IDXG=IDXG,
        num_warps=4,
    )

    M = triton.cdiv(T, BS)
    block_mask = torch.zeros(B, HI, T, M + 1, dtype=torch.bool, device=q.device)
    block_mask.scatter_(-1, sel.long().clamp(min=0).masked_fill(sel < 0, M), True)
    block_mask = block_mask[..., :M].contiguous()

    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    _gather_attn_bwd_dkv_kernel[(1, M, B * H)](
        q, k, v, lse, delta, do, dk, dv, block_mask, seg, scale,
        T=T, H=H, HQ=HQ, G=G, GP=GP, K=K, V=V, M=M, BS=BS, BK=BK, BV=BV,
        LW=LW, HI=HI, IDXG=IDXG,
        num_warps=4,
    )
    return dq, dk, dv


class _TritonGatherAttnFn(torch.autograd.Function):
    """Note: the decoder layer sits under gradient checkpointing, so the tensors
    saved here (incl. o/lse) only live transiently during that layer's recompute
    +backward window -- not across the whole model."""

    @staticmethod
    def forward(ctx, q, k, v, sel, seg, block_size, scale, local_window=None):
        o, lse = _gather_attn_fwd(q, k, v, sel, seg, block_size, scale, local_window)
        ctx.save_for_backward(q, k, v, sel, seg, o, lse)
        ctx.block_size = block_size
        ctx.scale = scale
        ctx.local_window = local_window
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, sel, seg, o, lse = ctx.saved_tensors
        dq, dk, dv = _gather_attn_bwd(
            q, k, v, o, lse, do.contiguous(), sel, seg, ctx.block_size, ctx.scale,
            ctx.local_window)
        return dq, dk, dv, None, None, None, None, None


def triton_gather_attention(
    query_states,                  
    key_states,                      
    value_states,                    
    block_indices,                                  
    position_ids,           
    attention_mask,
    block_size,
    scale,
    local_window=None,
):
    """Fused path entry. Returns [B, HQ, T, D], or None when this input shape is
    unsupported (caller must fall back to the eager gather path):
    padding/attention mask present, prefix kv cache (q_len != k_len), or
    index-head count not dividing the head layout. Supports HI >= 1 index heads
    ([B, HI, T, S]).

    supports sequence packing. A per-token seg_start (global start
    index of each token's sample segment) is derived as ``arange(T) - position_ids``
    (0 for a non-packed plain-arange row) and threaded into the kernels so every
    query only attends within [seg_start, t] -> no cross-sample leakage."""
    if attention_mask is not None:
        return None
    B, HQ, T, D = query_states.shape
    HI = block_indices.shape[1]
    Hkv = key_states.shape[1]
    if HI < 1 or HQ % HI != 0 or Hkv % HI != 0:
        return None
    if key_states.shape[2] != T:
        return None
    if position_ids is None or position_ids.dim() != 2 or position_ids.shape[-1] != T:
        return None
    ar = torch.arange(T, device=position_ids.device)
    seg = (ar[None, :] - position_ids).to(torch.int32).contiguous()

    sel = block_indices                 
    valid = sel >= 0
    sel_sorted, order = sel.sort(dim=-1)
    dup_sorted = torch.zeros_like(valid)
    dup_sorted[..., 1:] = sel_sorted[..., 1:] == sel_sorted[..., :-1]
    dup = torch.zeros_like(valid)
    dup.scatter_(-1, order, dup_sorted)
    sel = sel.masked_fill(dup & valid, -1)
    sel = sel.to(torch.int32).contiguous()

    q = query_states.transpose(1, 2).contiguous()                 
    k = key_states.transpose(1, 2).contiguous()                    
    v = value_states.transpose(1, 2).contiguous()
    o = _TritonGatherAttnFn.apply(q, k, v, sel, seg, block_size, scale, local_window)
    return o.transpose(1, 2)                 


@triton.jit
def _block_max_score_kernel(
    q_ptr, k_ptr, pos_ptr, seg_ptr, out_ptr,
    C, Sk,
    sqb, sqc, sqd, skb, sks, skd, spb, spc, sgb, sgc, sob, soc, son,
    D: tl.constexpr, BS: tl.constexpr, BC: tl.constexpr, BD: tl.constexpr,
    LW: tl.constexpr,
):
    i_c = tl.program_id(0)                
    i_blk = tl.program_id(1)             
    i_b = tl.program_id(2)           

    off_c = i_c * BC + tl.arange(0, BC)                    
    off_d = tl.arange(0, BD)                             
    m_c = off_c < C
    m_d = off_d < D

    b_q = tl.load(q_ptr + i_b * sqb + off_c[:, None] * sqc + off_d[None, :] * sqd,
                  mask=m_c[:, None] & m_d[None, :], other=0.0)
    b_pos = tl.load(pos_ptr + i_b * spb + off_c * spc, mask=m_c, other=0).to(tl.int32)
    b_seg = tl.load(seg_ptr + i_b * sgb + off_c * sgc, mask=m_c, other=0).to(tl.int32)
    b_ls = tl.maximum(b_pos - (LW - 1), 0)

    j = i_blk * BS + tl.arange(0, BS)                                      
    m_j = j < Sk
    b_k = tl.load(k_ptr + i_b * skb + j[:, None] * sks + off_d[None, :] * skd,
                  mask=m_j[:, None] & m_d[None, :], other=0.0)

    b_s = tl.dot(b_q, tl.trans(b_k))
    valid = (j[None, :] < b_ls[:, None]) & (j[None, :] >= b_seg[:, None]) & m_j[None, :]
    b_s = tl.where(valid, b_s, float('-inf'))
    b_max = tl.max(b_s, axis=1)                                     

    tl.store(out_ptr + i_b * sob + off_c * soc + i_blk * son, b_max, mask=m_c)


def triton_block_max_score(q, k, pos, seg, block_size, num_key_blocks, local_window=None):
    """Fused block-level max of q.k over NON-LOCAL causal SAME-SEGMENT keys; no dense
    [B,C,Sk] tensor.

    q:   [B, C, D]  (index-branch queries, single index head)
    k:   [B, Sk, D] (index-branch keys)
    pos: [B, C]     GLOBAL query positions
    seg: [B, C]     per-query segment start (0 for non-packed)
    Returns block_scores [B, C, num_key_blocks] float32 (-inf where a block has no
    valid key). Matches the eager `token_future | local_token | ~same_seg` masking."""
    assert q.dim() == 3 and k.dim() == 3, "expects [B, C, D] / [B, Sk, D]"
    B, C, D = q.shape
    Sk = k.shape[1]
    nblk = num_key_blocks
    BD = triton.next_power_of_2(D)
    if BD > 256:
        raise NotImplementedError(f"index dim D={D} > 256 not supported")
    BC = 64
    q = q.contiguous()
    k = k.contiguous()
    pos = pos.contiguous().to(torch.int32)
    seg = seg.contiguous().to(torch.int32)
    out = torch.empty(B, C, nblk, dtype=torch.float32, device=q.device)
    grid = (triton.cdiv(C, BC), nblk, B)
    _block_max_score_kernel[grid](
        q, k, pos, seg, out,
        C, Sk,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        pos.stride(0), pos.stride(1),
        seg.stride(0), seg.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        D=D, BS=block_size, BC=BC, BD=BD,
        LW=(block_size if local_window is None else max(1, int(local_window))),
        num_warps=4,
    )
    return out
