import allo
from allo.ir.types import int32
import numpy as np

def gemm(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
    C: int32[32, 32] = 0
    for i, j in allo.grid(32, 32):
        for k in allo.reduction(32):
         C[i, j] += A[i, k] * B[k, j]
    return C

s = allo.customize(gemm)
s.split("i", factor=8)
s.split("j", factor=8)
s.reorder("i.outer", "j.outer", "i.inner", "j.inner")
print(s.module)
mod = s.build(target="llvm")


# Prepare inputs using NumPy with appropriate data types
np_A = np.random.randint(0, 100, (32, 32)).astype(np.int32)
np_B = np.random.randint(0, 100, (32, 32)).astype(np.int32)

# Run the executable with the prepared inputs
np_C = mod(np_A, np_B)

# Perform a sanity check using NumPy's matmul as the golden reference
golden_C = np.matmul(np_A, np_B)

# Use numpy's testing framework to verify the correctness
np.testing.assert_allclose(np_C, golden_C, rtol=1e-5, atol=1e-5)

# Print success message if the results are correct
print("Results are correct!")