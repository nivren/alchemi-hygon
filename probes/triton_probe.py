"""Run only on one allocated HCU, with HIP_VISIBLE_DEVICES set."""
import json
import time
import torch
import triton
import triton.language as tl

@triton.jit
def add(X, Y, Z, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    tl.store(Z+i, tl.load(X+i, i<N, 0)+tl.load(Y+i, i<N, 0), i<N)

assert torch.version.hip and torch.cuda.is_available()
torch.manual_seed(42)
x = torch.randn(1025, device='cuda')
y = torch.randn_like(x)
z = torch.empty_like(x)
torch.cuda.synchronize()
t = time.perf_counter()
add[(triton.cdiv(x.numel(), 256),)](x, y, z, x.numel(), 256)
torch.cuda.synchronize()
cold = time.perf_counter()-t
torch.testing.assert_close(z, x+y, rtol=0, atol=0)
for _ in range(5): add[(5,)](x,y,z,1025,256)
torch.cuda.synchronize()
t = time.perf_counter()
for _ in range(20): add[(5,)](x,y,z,1025,256)
torch.cuda.synchronize()
print(json.dumps({'status':'passed','triton':triton.__version__,'file':triton.__file__, 'device':torch.cuda.get_device_name(), 'cold_seconds':cold,'warm_host_loop_seconds_per_call':(time.perf_counter()-t)/20, 'max_error':(z-x-y).abs().max().item()}))
