import allo
import numpy as np
from allo.ir.types import int32, float32
import time

np.random.seed(42)

def em(concrete_type, N):
    # Fix the kernel function type annotations
    def kernel_em[T: (float32, int32), N: int32](A: "T[N]", B: "T[N]", C: "T[N]"):
        for i in range(N):
            if A[i] == B[i]:
                C[i] = 1

    # Customize the kernel with concrete type and N
    s0 = allo.customize(kernel_em, instantiate=[concrete_type, N])
    s0.unroll('i',factor = 0)
    #s0.pipeline("i")
    
    # Return the built kernel function
    return s0.build(target="vitis_hls", mode="csim", project="exact_match_improved.prj")

# Python helper function for validation
def exact_match(A,B):
    # Determine the length of the shorter array
    min_len = min(len(A), len(B))

    # Initialize the output array to zero
    C = np.zeros(min_len, dtype=np.int32)

    # Compare elements up to the length of the shorter array
    for i in range(min_len):
        if A[i] == B[i]:
            C[i] = 1

    return C

def test_em():
    '''
    F = np.random.randint(low=10,high=20)
    G = np.random.randint(low=10,high=20)
    A = np.random.randint(size = F).astype(np.int32)
    B = np.random.randint(size = G).astype(np.int32)
    '''
    N = np.random.randint(low=10,high=20)
    concrete_type = int32
    mod = em(concrete_type, N)
    A = np.random.randint(low=0, high=100,size = N).astype(np.int32)
    B = np.random.randint(low=0, high=100,size = N).astype(np.int32)
    C = np.zeros(N, dtype=np.int32)
    A_ref = A.copy()
    B_ref = B.copy()
    C_ref = C.copy()
    C_ref=exact_match(A_ref, B_ref)
    mod(A,B,C)
    np.testing.assert_allclose(C, C_ref, rtol=1e-5, atol=1e-5)
    print("test passed")

# Call the Allo-generated function
test_em()
