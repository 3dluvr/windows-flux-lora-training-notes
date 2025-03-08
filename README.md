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

TBC

## Accelerate

TBA

## Triton

Another kind person has gone out of their way to keep Triton wheels for Windows up to date on their GH at https://github.com/woct0rdho/triton-windows/releases. Thank-you for doing it!
