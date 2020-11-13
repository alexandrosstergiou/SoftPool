import torch
import softpool_cuda
from SoftPool import soft_pool1d, soft_pool2d, soft_pool3d, SoftPool1d, SoftPool2d, SoftPool3d

import timeit


def check_close_enough(a, check):
    a = a.cpu()
    check = check.cpu()
    residual = (a-check).data.abs().mean().cpu().item()
    assert torch.isnan(check).sum() == 0, 'meet NaN(s) in `check`'
    assert residual < .2, 'residual is not small: {}'.format(residual)

x_1d = torch.rand((20, 32, 128)).float()
x_2d = torch.rand((20, 32, 128, 128)).float()
x_3d = torch.rand((20, 32, 16, 128, 128)).float()


print('\033[95m' + '--- Initial checks for forward ---' + '\033[0m')


################## 1D FORWARD ##################
print('\033[93m' + '> Checking 1D CPU ...' + '\033[0m')
try:
    pl_1d_cpu = soft_pool1d(x_1d)
    print('\033[92m' + '> PASSED' + '\033[0m')
except Exception as e:
    print('\033[91m' + '> FAILED' + '\033[0m')
    print(e)

print('\033[93m' + '> Checking 1D GPU ...' + '\033[0m')
try:
    pl_1d_gpu = soft_pool1d(x_1d.cuda())
    print('\033[92m' + '> PASSED' + '\033[0m')
except Exception as e:
    print('\033[91m' + '> FAILED' + '\033[0m')
    print(e)

print('\033[93m' + '> Checking 1D CPU-GPU output similarities ...' + '\033[0m')
try:
    check_close_enough(pl_1d_cpu.data, pl_1d_gpu.data)
    print('\033[92m' + '> PASSED' + '\033[0m'+'\n')
except Exception as e:
    print('\033[91m' + '> FAILED' + '\033[0m')
    print(e,'\n')

################## 2D FORWARD ##################
print('\033[93m' + '> Checking 2D CPU ...' + '\033[0m')
try:
    pl_2d_cpu = soft_pool2d(x_2d)
    print('\033[92m' + '> PASSED' + '\033[0m')
except Exception as e:
    print('\033[91m' + '> FAILED' + '\033[0m')
    print(e)

print('\033[93m' + '> Checking 2D GPU ...' + '\033[0m')
try:
    pl_2d_gpu = soft_pool2d(x_2d.cuda())
    print('\033[92m' + '> PASSED' + '\033[0m')
except Exception as e:
    print('\033[91m' + '> FAILED' + '\033[0m')
    print(e)

print('\033[93m' + '> Checking 2D CPU-GPU output similarities ...' + '\033[0m')
try:
    check_close_enough(pl_2d_cpu.data, pl_2d_gpu.data)
    print('\033[92m' + '> PASSED' + '\033[0m'+'\n')
except Exception as e:
    print('\033[91m' + '> FAILED' + '\033[0m')
    print(e,'\n')

################## 3D FORWARD ##################
print('\033[93m' + '> Checking 3D CPU ...' + '\033[0m')
try:
    pl_3d_cpu = soft_pool3d(x_3d)
    print('\033[92m' + '> PASSED' + '\033[0m')
except Exception as e:
    print('\033[91m' + '> FAILED' + '\033[0m')
    print(e)

print('\033[93m' + '> Checking 3D GPU ...' + '\033[0m')
try:
    pl_3d_gpu = soft_pool3d(x_3d.cuda())
    print('\033[92m' + '> PASSED' + '\033[0m')
except Exception as e:
    print('\033[91m' + '> FAILED' + '\033[0m')
    print(e)

print('\033[93m' + '> Checking 3D CPU-GPU output similarities ...' + '\033[0m')
try:
    check_close_enough(pl_3d_cpu.data, pl_3d_gpu.data)
    print('\033[92m' + '> PASSED' + '\033[0m'+'\n')
except Exception as e:
    print('\033[91m' + '> FAILED' + '\033[0m')
    print(e,'\n')


print('\033[95m' + '--- Initial checks for backward ---' + '\033[0m')

a_1d = torch.rand((20, 32, 128)).float()
b_1d = a_1d.clone().cuda()
a_2d = torch.rand((20, 32, 128, 128)).float()
b_2d = a_2d.clone().cuda()
a_3d = torch.rand((20, 32, 16, 128, 128)).float()
b_3d = a_3d.clone().cuda()

a_1d.requires_grad = True
a_2d.requires_grad = True
a_3d.requires_grad = True
b_1d.requires_grad = True
b_2d.requires_grad = True
b_3d.requires_grad = True


print('\033[93m' + '> Checking 1D CPU ...' + '\033[0m')
try:
    soft_pool1d(a_1d).pow(2).mean().backward()
    print('\033[92m' + '> PASSED' + '\033[0m')
except Exception as e:
    print('\033[91m' + '> FAILED' + '\033[0m')
    print(e)

print('\033[93m' + '> Checking 1D GPU ...' + '\033[0m')
try:
    soft_pool1d(b_1d).pow(2).mean().backward()
    print('\033[92m' + '> PASSED' + '\033[0m')
except Exception as e:
    print('\033[91m' + '> FAILED' + '\033[0m')
    print(e)

print('\033[93m' + '> Checking 1D grad similarities ...' + '\033[0m')
try:
    check_close_enough(a_1d.grad.data, b_1d.grad.data)
    print('\033[92m' + '> PASSED' + '\033[0m'+'\n')
except Exception as e:
    print('\033[91m' + '> FAILED' + '\033[0m')
    print(e,'\n')

print('\033[93m' + '> Checking 2D CPU ...' + '\033[0m')
try:
    soft_pool2d(a_2d).pow(2).mean().backward()
    print('\033[92m' + '> PASSED' + '\033[0m')
except Exception as e:
    print('\033[91m' + '> FAILED' + '\033[0m')
    print(e)

print('\033[93m' + '> Checking 2D GPU ...' + '\033[0m')
try:
    soft_pool2d(b_2d).pow(2).mean().backward()
    print('\033[92m' + '> PASSED' + '\033[0m')
except Exception as e:
    print('\033[91m' + '> FAILED' + '\033[0m')
    print(e)

print('\033[93m' + '> Checking 2D grad similarities ...' + '\033[0m')
try:
    check_close_enough(a_2d.grad.data, b_2d.grad.data)
    print('\033[92m' + '> PASSED' + '\033[0m'+'\n')
except Exception as e:
    print('\033[91m' + '> FAILED' + '\033[0m')
    print(e,'\n')

print('\033[93m' + '> Checking 3D CPU ...' + '\033[0m')
try:
    soft_pool3d(a_3d).pow(2).mean().backward()
    print('\033[92m' + '> PASSED' + '\033[0m')
except Exception as e:
    print('\033[91m' + '> FAILED' + '\033[0m')
    print(e)

print('\033[93m' + '> Checking 3D GPU ...' + '\033[0m')
try:
    soft_pool3d(b_3d).pow(2).mean().backward()
    print('\033[92m' + '> PASSED' + '\033[0m')
except Exception as e:
    print('\033[91m' + '> FAILED' + '\033[0m')
    print(e)

print('\033[93m' + '> Checking 3D grad similarities ...' + '\033[0m')
try:
    check_close_enough(a_3d.grad.data, b_3d.grad.data)
    print('\033[92m' + '> PASSED' + '\033[0m'+'\n')
except Exception as e:
    print('\033[91m' + '> FAILED' + '\033[0m')
    print(e,'\n')


print('\n'+'\033[92m' + 'TESTS COMPLETED' + '\033[0m'+'\n')

print('\033[95m' + '--- Profiling checks ---' + '\033[0m')

a_1d = torch.rand((10, 32, 80)).float()
b_1d = a_1d.clone().cuda()
c_1d = a_1d.clone().cuda()
a_2d = torch.rand((10, 32, 80, 80)).float()
b_2d = a_2d.clone().cuda()
c_2d = a_2d.clone().cuda()
a_3d = torch.rand((10, 32, 8, 80, 80)).float()
b_3d = a_3d.clone().cuda()
c_3d = a_3d.clone().cuda()


a_1d.requires_grad = True
a_2d.requires_grad = True
a_3d.requires_grad = True
b_1d.requires_grad = True
b_2d.requires_grad = True
b_3d.requires_grad = True
c_1d.requires_grad = True
c_2d.requires_grad = True
c_3d.requires_grad = True


with torch.autograd.profiler.profile(use_cuda=False) as prof:
    for i in range(100):
        soft_pool1d(a_1d)
print('\033[93m' +'SoftPool1d (CPU) [foward]'+ '\033[0m')
print(prof.key_averages().table(sort_by="self_cpu_time_total"))
time_f_1d_cpu = ''.join(str(prof).split('\n')[-2:])
_tt = soft_pool1d(a_1d)
with torch.autograd.profiler.profile(use_cuda=False) as prof:
    for i in range(100):
        soft_pool1d(a_1d).backward(_tt)
print('\033[93m' +'SoftPool1d (CPU) [forward + backward]'+ '\033[0m')
print(prof.key_averages().table(sort_by="self_cpu_time_total"))
time_b_1d_cpu = ''.join(str(prof).split('\n')[-2:])

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(100):
        soft_pool1d(b_1d,force_inplace=True)
print('\033[93m' +'SoftPool1d (CUDA-inplace) [foward]'+ '\033[0m')
print(prof.key_averages())
time_f_1d_cuda_forced = ''.join(str(prof).split('\n')[-3:])
_tt = soft_pool1d(b_1d,force_inplace=True)
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(100):
        soft_pool1d(b_1d,force_inplace=True).backward(_tt)
print('\033[93m' +'SoftPool1d (CUDA-inplace) [forward + backward]'+ '\033[0m')
print(prof.key_averages())
time_b_1d_cuda_forced = ''.join(str(prof).split('\n')[-3:])

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(100):
        soft_pool1d(c_1d)
print('\033[93m' +'SoftPool1d (CUDA) [foward]'+ '\033[0m')
print(prof.key_averages())
time_f_1d_cuda = ''.join(str(prof).split('\n')[-3:])
_tt = soft_pool1d(c_1d)
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(100):
        soft_pool1d(c_1d).backward(_tt)
print('\033[93m' +'SoftPool1d (CUDA) [forward + backward]'+ '\033[0m')
print(prof.key_averages())
time_b_1d_cuda = ''.join(str(prof).split('\n')[-3:])



with torch.autograd.profiler.profile(use_cuda=False) as prof:
    for i in range(100):
        soft_pool2d(a_2d)
print('\033[93m' +'SoftPool2d (CPU) [foward]'+ '\033[0m')
print(prof.key_averages().table(sort_by="self_cpu_time_total"))
time_f_2d_cpu = ''.join(str(prof).split('\n')[-2:])
_tt = soft_pool2d(a_2d)
with torch.autograd.profiler.profile(use_cuda=False) as prof:
    for i in range(100):
        soft_pool2d(a_2d).backward(_tt)
print('\033[93m' +'SoftPool2d (CPU) [forward + backward]'+ '\033[0m')
print(prof.key_averages().table(sort_by="self_cpu_time_total"))
time_b_2d_cpu = ''.join(str(prof).split('\n')[-2:])

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(100):
        soft_pool2d(b_2d,force_inplace=True)
print('\033[93m' +'SoftPool2d (CUDA-inplace) [foward]'+ '\033[0m')
print(prof.key_averages())
time_f_2d_cuda_forced = ''.join(str(prof).split('\n')[-3:])
_tt = soft_pool2d(b_2d,force_inplace=True)
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(100):
        soft_pool2d(b_2d,force_inplace=True).backward(_tt)
print('\033[93m' +'SoftPool2d (CUDA-inplace) [forward + backward]'+ '\033[0m')
print(prof.key_averages())
time_b_2d_cuda_forced = ''.join(str(prof).split('\n')[-3:])

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(100):
        soft_pool2d(c_2d)
print('\033[93m' +'SoftPool2d (CUDA) [foward]'+ '\033[0m')
time_f_2d_cuda = ''.join(str(prof).split('\n')[-3:])
print(prof.key_averages())
_tt = soft_pool2d(c_2d)
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(100):
        soft_pool2d(c_2d).backward(_tt)
print('\033[93m' +'SoftPool2d (CUDA) [forward + backward]'+ '\033[0m')
print(prof.key_averages())
time_b_2d_cuda = ''.join(str(prof).split('\n')[-3:])



with torch.autograd.profiler.profile(use_cuda=False) as prof:
    for i in range(100):
        soft_pool3d(a_3d)
print('\033[93m' +'SoftPool3d (CPU) [foward]'+ '\033[0m')
print(prof.key_averages().table(sort_by="self_cpu_time_total"))
time_f_3d_cpu = ''.join(str(prof).split('\n')[-2:])
_tt = soft_pool3d(a_3d)
with torch.autograd.profiler.profile(use_cuda=False) as prof:
    for i in range(100):
        soft_pool3d(a_3d).backward(_tt)
print('\033[93m' +'SoftPool3d (CPU) [forward + backward]'+ '\033[0m')
print(prof.key_averages().table(sort_by="self_cpu_time_total"))
time_b_3d_cpu = ''.join(str(prof).split('\n')[-2:])

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(100):
        soft_pool3d(b_3d,force_inplace=True)
print('\033[93m' +'SoftPool3d (CUDA-inplace) [foward]'+ '\033[0m')
print(prof.key_averages())
time_f_3d_cuda_forced = ''.join(str(prof).split('\n')[-3:])
_tt = soft_pool3d(b_3d,force_inplace=True)
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(100):
        soft_pool3d(b_3d,force_inplace=True).backward(_tt)
print('\033[93m' +'SoftPool3d (CUDA-inplace) [forward + backward]'+ '\033[0m')
print(prof.key_averages())
time_b_3d_cuda_forced = ''.join(str(prof).split('\n')[-3:])

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(100):
        soft_pool3d(c_3d)
print('\033[93m' +'SoftPool3d (CUDA) [foward]'+ '\033[0m')
print(prof.key_averages())
time_f_3d_cuda = ''.join(str(prof).split('\n')[-3:])
_tt = soft_pool3d(c_3d)
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(100):
        soft_pool3d(c_3d).backward(_tt)
print('\033[93m' +'SoftPool3d (CUDA) [forward + backward]'+ '\033[0m')
print(prof.key_averages())
time_b_3d_cuda = ''.join(str(prof).split('\n')[-3:])


print('\n'+'\033[93m' +'-------------------------------'+ '\033[0m')
print('\033[93m' +'SoftPool1d [forward + backward]'+ '\033[0m')
print('\n'+'\033[93m' +'----------- C P U ------------'+ '\033[0m')
print(time_b_1d_cpu)
print('\n'+'\033[93m' +'-- C U D A - I N P L A C E ---'+ '\033[0m')
print(time_b_1d_cuda_forced)
print('\n'+'\033[93m' +'---------- C U D A -----------'+ '\033[0m')
print(time_b_1d_cuda)
print('\n'+'\033[93m' +'-------------------------------'+ '\033[0m')

print('\n'+'\033[93m' +'-------------------------------'+ '\033[0m')
print('\033[93m' +'SoftPool2d [forward + backward]'+ '\033[0m')
print('\n'+'\033[93m' +'----------- C P U ------------'+ '\033[0m')
print(time_b_2d_cpu)
print('\n'+'\033[93m' +'-- C U D A - I N P L A C E ---'+ '\033[0m')
print(time_b_2d_cuda_forced)
print('\n'+'\033[93m' +'---------- C U D A -----------'+ '\033[0m')
print(time_b_2d_cuda)
print('\n'+'\033[93m' +'-------------------------------'+ '\033[0m')

print('\n'+'\033[93m' +'-------------------------------'+ '\033[0m')
print('\033[93m' +'SoftPool3d [forward + backward]'+ '\033[0m')
print('\n'+'\033[93m' +'----------- C P U ------------'+ '\033[0m')
print(time_b_3d_cpu)
print('\n'+'\033[93m' +'-- C U D A - I N P L A C E ---'+ '\033[0m')
print(time_b_3d_cuda_forced)
print('\n'+'\033[93m' +'---------- C U D A -----------'+ '\033[0m')
print(time_b_3d_cuda)
print('\n'+'\033[93m' +'-------------------------------'+ '\033[0m')

print('\n'+'\033[95m' + '--- Tests finished ---' + '\033[0m')
