# Windows FLUX LoRA Training Notes
An ever growing collection of random notes on training FLUX LoRA in Window, and resolving various issues along the way. The main idea here is not to fix any of the dependent modules, but to get going fast and without delay.

- In general training in Windows is an utter hit-and-miss because major Python modules are not coded with Windows in mind (Accelerate, PyTorch, Flash Attention...);
- Some people say that Kohya-SS sd-scripts for training Flux are not working correctly; this also means anything using them (Fluxgym, etc.) will also not work well. At least one thing needs to be fixed in ```train_util.py:L5353``` (the Version check) so the line reads ```"env://?use_libuv=False" if os.name == "nt" and torch.__version__ >= "2.4.0" else None```
- From my tests, I'm getting better results with Kohya-SS sd-scripts than with OneTrainer. Bonus with sd-scripts is that through Accelerate it uses distributed torch which allows training on more than one GPU. So far I haven't seen OneTrainer using more than one GPU in my system.
- Not all models are made the same, and not all conversions worked out well. Yet, there are many copies of the copies that are badly done and finding the right model that actually works correctly (aligns with the original) is difficult. There are many de-distillations and abliterations and they all claim one thing or another.

## PyTorch

Many people think there's some kind of a secret formula/mix of modules that will get their setup working. They resort to installing old versions of PyTorch and CUDA (e.g. torch-2.3.1+cu118 from https://pytorch.org/get-started/previous-versions/) because they say that's where it works and after that it doesn't. The confusion comes from the fact that as these modules were being worked on by the devs and many bugs fixed, further divergence from Windows had occurred. Unfortunately, the only way to fix them is to patch the code and force certain things to play nice in Windows.

Most recent versions have advanced the distributed backend (Rendezvous) in favour of the old c10d (now legacy), which apparently did work in Windows at some point in the past. Thus those people mentioned above suggesting the use of torch-2.3.1+cu118 for example.

Putting *set USE_LIBUV=0* in your venv is meaningless because it only applies during compilation of PyTorch, and compiling it in Windows is a real pain in the butt. So you still get that ```RuntimeError: use_libuv was requested but PyTorch was build without libuv support```

### patched files (this probably only applies to distributed training with mutliple GPUs in the system)

Accelerate needs to pass the ```backend="gloo"``` for Windows somewhere, but for the time being

Lib\site-packages\torch\distributed\distributed_c10d.py:L1674 we'll force it
```
if os.name == "nt":
    backend = "gloo"
```
Then...

Lib\site-packages\torch\distributed\rendezvous.py

TCPStore( ... needs ```use_libuv=False,```

Lib\site-packages\torch\distributed\elastic\rendezvous\static_tcp_rendezvous.py

TCPStore( ... needs ```use_libuv=False,```

Patching is needed because the URL does not appear to be exposed anywhere so one can't add the request parameter use_libuv=False to it. master_port and master_host can be set from the environment, but that's about it.

To test Distributed Processing in Windows, easiest sample code would be:

```
import os
import torch.distributed as dist

#dist_url = 'tcp://127.0.0.1:23456?use_libuv=False'

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '23456'
dist_url = 'env://?use_libuv=False'

#dist_url = 'file:///c:/temp/torch_dp'

dist.init_process_group(
   backend = 'gloo',
   rank=0,
   init_method=dist_url,
   world_size=1)
print("DP is working")
```
Uncommenting one of the lines/blocks that describe the ```dist_url```, obviously ```tcp://``` and ```env://``` are similar/identical, except that the addr/port come from different sources. The ```file://``` might be the most ideal one, although there are suggestions not to use it (why not?) The key is the **backend** which needs to be ```gloo``` because ```nccl``` isn't supported in Windows.

More on the TCP backend with Libuv at https://pytorch.org/tutorials/intermediate/TCPStore_libuv_backend.html

## Flash Attention with CUDA

Another person has kindly taken on a task of providing pre-compiled wheels for flash_attn module with CUDA on their GH at https://github.com/kingbri1/flash-attention/releases. The releases there include major Python versions, and the last several PyTorch versions. Thank-you for doing it!

*set USE_FLASH_ATTENTION=1* does work when you set it in your venv, and install the above mentioned flash_attn module with CUDA compiled in. But that's not enough because ```flash_attn_2_cuda.cp312-win_amd64.pyd``` still fails to find deps and throws a fit: ```DLL load failed while importing flash_attn_2_cuda: The specified module could not be found.```

That .pyd depends on the following dlls:
```
api-ms-win-crt-heap-l1-1-0.dll       Loaded successfully
api-ms-win-crt-math-l1-1-0.dll       Loaded successfully
api-ms-win-crt-runtime-l1-1-0.dll    Loaded successfully
api-ms-win-crt-stdio-l1-1-0.dll      Loaded successfully
api-ms-win-crt-string-l1-1-0.dll     Loaded successfully
c10.dll                              Error 126: The specified module could not be found.
c10_cuda.dll                         Error 126: The specified module could not be found.
cudart64_12.dll                      Loaded successfully
KERNEL32.dll                         Loaded successfully
MSVCP140.dll                         Loaded successfully
python312.dll                        Loaded successfully
torch_cpu.dll                        Error 126: The specified module could not be found.
torch_cuda.dll                       Error 126: The specified module could not be found.
torch_python.dll                     Error 126: The specified module could not be found.
VCRUNTIME140.dll                     Loaded successfully
VCRUNTIME140_1.dll                   Loaded successfully
```
So, more patching is needed, inside ```Lib\site-packages\flash_attnflash_attn_interface.py``` It needs to know where to find those missing dlls (they come from PyTorch). Before Python 3.8, one could simply add the paths to the system, but not anymore - now have to use ```os.add_dll_directory(path)``` for that purpose. The paths that need adding are to PyTorch **lib** folder as well as **Library\bin** folder of your venv.

#### Compiling

```
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git submodule update --init --recursive
python setup.py bdist_wheel
or
pip wheel . -w wheels
```

TBC

## Accelerate

TBA

## Triton

Another kind person has gone out of their way to keep Triton wheels for Windows up to date on their GH at https://github.com/woct0rdho/triton-windows/releases. Thank-you for doing it!

## DeepSpeed

I compiled DeepSpeed 0.16.4 for Python3.12 with CUDA 12.4, PyTorch 2.6.0 and Triton 3.1.0...it seems to be ok based on the ```ds_report```

```
[2025-03-08 19:58:11,587] [INFO] [real_accelerator.py:222:get_accelerator] Setting ds_accelerator to cuda (auto detect)
test.c
LINK : fatal error LNK1181: cannot open input file 'aio.lib'
test.c
LINK : fatal error LNK1181: cannot open input file 'cufile.lib'
W0308 19:58:15.602000 3136 Lib\site-packages\torch\distributed\elastic\multiprocessing\redirects.py:29] NOTE: Redirects are currently not supported in Windows or MacOs.
--------------------------------------------------
DeepSpeed C++/CUDA extension op report
--------------------------------------------------
NOTE: Ops not installed will be just-in-time (JIT) compiled at
      runtime if needed. Op compatibility means that your system
      meet the required dependencies to JIT install the op.
--------------------------------------------------
JIT compiled ops requires ninja
ninja .................. [OKAY]
--------------------------------------------------
op name ................ installed .. compatible
--------------------------------------------------
test.c
LINK : fatal error LNK1181: cannot open input file 'aio.lib'
 [WARNING]  async_io requires the dev libaio .so object and headers but these were not found.
 [WARNING]  If libaio is already installed (perhaps from source), try setting the CFLAGS and LDFLAGS environment variables to where it can be found.
async_io ............... [NO] ....... [NO]
fused_adam ............. [YES] ...... [OKAY]
cpu_adam ............... [YES] ...... [OKAY]
cpu_adagrad ............ [YES] ...... [OKAY]
cpu_lion ............... [YES] ...... [OKAY]
 [WARNING]  Please specify the CUTLASS repo directory as environment variable $CUTLASS_PATH
evoformer_attn ......... [NO] ....... [NO]
 [WARNING]  FP Quantizer is using an untested triton version (3.1.0), only 2.3.(0, 1) and 3.0.0 are known to be compatible with these kernels
fp_quantizer ........... [NO] ....... [NO]
fused_lamb ............. [YES] ...... [OKAY]
fused_lion ............. [YES] ...... [OKAY]
test.c
LINK : fatal error LNK1181: cannot open input file 'cufile.lib'
gds .................... [NO] ....... [NO]
transformer_inference .. [YES] ...... [OKAY]
inference_core_ops ..... [YES] ...... [OKAY]
cutlass_ops ............ [NO] ....... [OKAY]
quantizer .............. [YES] ...... [OKAY]
ragged_device_ops ...... [NO] ....... [OKAY]
ragged_ops ............. [YES] ...... [OKAY]
random_ltd ............. [YES] ...... [OKAY]
 [WARNING]  sparse_attn requires a torch version >= 1.5 and < 2.0 but detected 2.6
 [WARNING]  using untested triton version (3.1.0), only 1.0.0 is known to be compatible
sparse_attn ............ [NO] ....... [NO]
spatial_inference ...... [YES] ...... [OKAY]
transformer ............ [YES] ...... [OKAY]
stochastic_transformer . [YES] ...... [OKAY]
--------------------------------------------------
DeepSpeed general environment info:
torch install path ............... ['G:\\FluxGym\\Lib\\site-packages\\torch']
torch version .................... 2.6.0+cu124
deepspeed install path ........... ['G:\\FluxGym\\Lib\\site-packages\\deepspeed']
deepspeed info ................... 0.16.4+unknown, unknown, unknown
torch cuda version ............... 12.4
torch hip version ................ None
nvcc version ..................... 12.4
deepspeed wheel compiled w. ...... torch 2.6, cuda 12.4
shared memory (/dev/shm) size .... UNKNOWN
```

Uploaded **deepspeed-0.16.4+cu124torch2.6.0-cp312-cp312-win_amd64.whl** in the Releases of this repo. You must have CUDA 124 toolkit installed, etc.
