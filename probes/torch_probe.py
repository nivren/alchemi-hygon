"""Explicit CPU or allocated DCU smoke; never automatically falls back."""
import argparse
import json
import importlib.metadata
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument('--device', choices=['cpu', 'cuda'], required=True)
a = p.parse_args()
if a.device == 'cuda':
    assert torch.version.hip and torch.cuda.is_available(), 'HIP device required'
array = np.array([1., 2.], dtype=np.float64)
assert np.array_equal(torch.from_numpy(array).numpy(), array)
torch.manual_seed(42)
x = torch.tensor([1., 2., 3.], dtype=torch.float64, device=a.device, requires_grad=True)
g = torch.autograd.grad((x**3).sum(), x, create_graph=True)[0]
h = torch.autograd.grad(g.sum(), x)[0]
torch.testing.assert_close(g, 3*x*x, rtol=1e-12, atol=1e-12)
torch.testing.assert_close(h, 6*x, rtol=1e-12, atol=1e-12)
idx = torch.tensor([0, 1, 0], device=a.device)
y = torch.zeros(2, dtype=x.dtype, device=a.device).index_add(0, idx, x)
torch.testing.assert_close(y, x.new_tensor([4., 2.]), rtol=0, atol=0)
sg = torch.autograd.grad(y.square().sum(), x, create_graph=True)[0]
sh = torch.autograd.grad(sg.sum(), x)[0]
torch.testing.assert_close(sh, x.new_tensor([4., 2., 4.]), rtol=0, atol=0)
z = torch.fft.ifft(torch.fft.fft(x))
torch.testing.assert_close(z.real, x, rtol=1e-12, atol=1e-12)
if a.device == 'cuda': torch.cuda.synchronize()
print(json.dumps({'torch': torch.__version__, 'torch_distribution': importlib.metadata.version('torch'), 'numpy': np.__version__, 'file': torch.__file__, 'hip': torch.version.hip,
                  'device': a.device, 'dtype': str(x.dtype), 'seed': 42, 'gradient': g.tolist(),
                  'second_gradient': h.tolist(), 'segment': y.tolist(), 'fft_max_error': (z.real-x).abs().max().item(), 'status': 'passed'}))
