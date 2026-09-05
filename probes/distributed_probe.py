"""Exactly two allocated HCUs: torchrun --standalone --nproc-per-node=2."""
import datetime
import json
import os
import torch
import torch.distributed as dist

assert torch.version.hip
rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=45))
assert dist.get_world_size() == 2
try:
    x = torch.tensor([rank+1.], device='cuda')
    dist.all_reduce(x)
    assert x.item() == 3
    send = torch.tensor([rank+7.], device='cuda')
    recv = torch.empty_like(send)
    reqs = dist.batch_isend_irecv([dist.P2POp(dist.isend,send,1-rank), dist.P2POp(dist.irecv,recv,1-rank)])
    for r in reqs: r.wait()
    torch.cuda.synchronize()
    assert recv.item() == 8-rank
    print(json.dumps({'rank':rank,'all_reduce':x.item(),'received':recv.item(),'status':'passed'}))
finally:
    dist.destroy_process_group()
