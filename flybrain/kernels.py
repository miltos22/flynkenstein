import weakref
import torch
import triton
import triton.language as tl
RING = 64

@triton.jit
def _graded_rows(row_ptr, pre, delay, ch_f, w_f, w_s, ring, out, n_step, N, BLOCK: tl.constexpr):
    i = tl.program_id(0)
    st = tl.load(row_ptr + i)
    en = tl.load(row_ptr + i + 1)
    a_ef = tl.zeros([BLOCK], dtype=tl.float32)
    a_es = tl.zeros([BLOCK], dtype=tl.float32)
    a_if = tl.zeros([BLOCK], dtype=tl.float32)
    a_is = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(st, en, BLOCK):
        e = off + tl.arange(0, BLOCK)
        m = e < en
        p = tl.load(pre + e, mask=m, other=0)
        d = tl.load(delay + e, mask=m, other=0)
        slot = (n_step - d + 64 * 64) % 64
        src = tl.load(ring + slot.to(tl.int64) * N + p, mask=m, other=0.0)
        c = tl.load(ch_f + e, mask=m, other=0)
        vf = tl.load(w_f + e, mask=m, other=0.0) * src
        vs = tl.load(w_s + e, mask=m, other=0.0) * src
        exc = c == 0
        a_ef += tl.where(exc, vf, 0.0)
        a_es += tl.where(exc, vs, 0.0)
        a_if += tl.where(exc, 0.0, vf)
        a_is += tl.where(exc, 0.0, vs)
    tl.store(out + i, tl.load(out + i) + tl.sum(a_ef))
    tl.store(out + N + i, tl.load(out + N + i) + tl.sum(a_es))
    tl.store(out + 2 * N + i, tl.load(out + 2 * N + i) + tl.sum(a_if))
    tl.store(out + 3 * N + i, tl.load(out + 3 * N + i) + tl.sum(a_is))

@triton.jit
def _spike_scatter(tilde, is_graded, out_ptr, delay, idx_f, idx_s, w_f, w_s, buf, n_step, N4, BLOCK: tl.constexpr):
    j = tl.program_id(0)
    t = tl.load(tilde + j)
    g = tl.load(is_graded + j)
    if (t != 0.0) & (g == 0):
        st = tl.load(out_ptr + j)
        en = tl.load(out_ptr + j + 1)
        for off in range(st, en, BLOCK):
            e = off + tl.arange(0, BLOCK)
            m = e < en
            d = tl.load(delay + e, mask=m, other=0)
            base = ((n_step + d) % 64).to(tl.int64) * N4
            tl.atomic_add(buf + base + tl.load(idx_f + e, mask=m, other=0), tl.load(w_f + e, mask=m, other=0.0) * t, mask=m)
            tl.atomic_add(buf + base + tl.load(idx_s + e, mask=m, other=0), tl.load(w_s + e, mask=m, other=0.0) * t, mask=m)

class FastTransmission:

    def __init__(self, net):
        N = net.N
        gd, sp = (net.gd, net.sp)
        post = gd['idx_f'] % N
        order = torch.argsort(post, stable=True)
        self.g = {k: gd[k][order] for k in ('pre', 'delay', 'idx_f', 'w_f', 'w_s')}
        cnt = torch.bincount(post[order], minlength=N)
        self.g_row = torch.cat([torch.zeros(1, dtype=torch.long, device=post.device), torch.cumsum(cnt, 0)]).to(torch.int64)
        self.g_ch = (self.g['idx_f'] // N).to(torch.int32)
        self.g_pre = self.g['pre'].to(torch.int32)
        self.g_delay = self.g['delay'].to(torch.int32)
        self.s_delay = sp['delay'].to(torch.int32)
        self.s_idx_f, self.s_idx_s = (sp['idx_f'], sp['idx_s'])
        self.net = weakref.proxy(net)

    def graded(self, out, ring, n):
        N = self.net.N
        _graded_rows[N,](self.g_row, self.g_pre, self.g_delay, self.g_ch, self.g['w_f'], self.g['w_s'], ring, out, n, N, BLOCK=32)

    def scatter(self, tilde, is_graded_u8, n):
        net = self.net
        _spike_scatter[net.N,](tilde, is_graded_u8, net.out_ptr, self.s_delay, self.s_idx_f, self.s_idx_s, net.sp['w_f'], net.sp['w_s'], net.buf, n, 4 * net.N, BLOCK=64)
