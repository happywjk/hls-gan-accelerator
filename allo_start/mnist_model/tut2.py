import allo
from allo.ir.types import float32
M, N, K = 1024, 1024, 1024
def gemm(A: float32[M, K], B: float32[K, N]) -> float32[M, N]:
    C: float32[M, N] = 0.0
    for i, j in allo.grid(M, N):
        for k in allo.reduction(K):
            C[i, j] += A[i, k] * B[k, j]
    return C

s = allo.customize(gemm)
s.reorder("k", "j")
s.buffer_at(s.C, axis="i")
s.pipeline("j")
code = s.build(target="vhls")
print(code)
mod = s.build(target="vitis_hls", mode="csyn", project="gemm.prj")
mod()