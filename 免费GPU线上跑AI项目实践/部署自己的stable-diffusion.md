# 部署自己的stable-diffusion

###


公开镜像:AUTOMATIC1111/stable-diffusion-webui
数据集：stable-diffusion-models

![](/img/Stable-diffusion/0-创建项目.png)

![](/img/Stable-diffusion//0-1初始化开发环境.png)
运行初始化中，等待约 5-10 分钟，当右侧的 网页终端 和 JupyterLab 不再是灰色时，表明工具注入成功。此时您便可在此开发环境上通过工具进行模型调优，详情可参见下一步。
![](/img/Stable-diffusion/0-1初始化开发环境-2.png)

依次输入以下4串命令就可以啦，注意每一次命令输入之后要等它运行完毕之后再输入下一条指令

1、解压代码及模型 tar xf /gemini/data-1/stable-diffusion-webui.tar -C /gemini/code/ 

~~2、解压配置文件到隐藏目录/root/.cache~~
~~tar xf /gemini/data-1/cache.tar -C /root/~~ 
~~3、拷贝frpc内网穿透文件 （注意有两行 -> 两条指令）~~
~~cp /gemini/data-1/frpc_linux_amd64 /root/miniconda3/lib/python3.10/site-packages/gradio/frpc_linux_amd64_v0.2%E2%80%8B%0D%0Achmod +x /root/miniconda3/lib/python3.10/site-packages/gradio/frpc_linux_amd64_v0.2~~
###### 新版改过了 .cache frpc已经有了 指令2,3 不用再执行
![](/img/Stable-diffusion/1-cmd-pre-error-cache.tar.png)
![](/img/Stable-diffusion/1-cmd-pre-cache-frpc-pass.png)
4、运行项目 （注意有两行 -> 两条指令）
cd /gemini/code/stable-diffusion-webui
python launch.py --deepdanbooru --theme dark --xformers --listen --gradio-auth dolan:123456
![](/img/Stable-diffusion/1-cmd-pre-error-cache.tar.png)


``` shell
(base) root@375877374027239424-taskrole1-0:/gemini/code/stable-diffusion-webui# python launch.py --deepdanbooru --theme dark --xformers --listen --gradio-auth dolan:123456
Python 3.10.10 (main, Mar 21 2023, 18:45:11) [GCC 11.2.0]
Version: v1.6.0
Commit hash: 5ef669de080814067961f28357256e8fe27544f4
Installing requirements for CodeFormer
Launching Web UI with arguments: --deepdanbooru --theme dark --xformers --listen --gradio-auth dolan:123456

===================================BUG REPORT===================================
Welcome to bitsandbytes. For bug reports, please run

python -m bitsandbytes

 and submit this information together with your error trace to: https://github.com/TimDettmers/bitsandbytes/issues
================================================================================
bin /root/miniconda3/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cuda118.so
/root/miniconda3/lib/python3.10/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: /root/miniconda3 did not contain ['libcudart.so', 'libcudart.so.11.0', 'libcudart.so.12.0'] as expected! Searching further paths...
  warn(msg)
/root/miniconda3/lib/python3.10/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: WARNING: The following directories listed in your path were found to be non-existent: {PosixPath('/usr/lib/orion')}
  warn(msg)
/root/miniconda3/lib/python3.10/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: /opt/orion/orion_runtime/lib:/usr/lib64:/usr/lib:/usr/lib/orion did not contain ['libcudart.so', 'libcudart.so.11.0', 'libcudart.so.12.0'] as expected! Searching further paths...
  warn(msg)
/root/miniconda3/lib/python3.10/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: WARNING: The following directories listed in your path were found to be non-existent: {PosixPath('"wqubxwo3rzmo/1313-20230915134904'), PosixPath('[{"ext_type"'), PosixPath('"data-1","type"'), PosixPath('4,"cpuCommitCent"'), PosixPath('"1313","data_id"'), PosixPath('"8888"}}]}'), PosixPath('{"data_type"'), PosixPath('{"type"'), PosixPath('"code","type"'), PosixPath('{"cpu"'), PosixPath('"NVIDIA GeForce RTX 3090","count"'), PosixPath('[{"name"'), PosixPath('"weed","path"'), PosixPath('"Codeset"},{"name"'), PosixPath('1,"image"'), PosixPath('22}],"extensions"'), PosixPath('\\"seaweedfs-filer.gemini-storage'), PosixPath('"/pavostor/gemini/codeset/rg1oriahgqk1/375876958015197184/latest","read_only"'), PosixPath('22602\\"}","data_location"'), PosixPath('5004","username"'), PosixPath('{"image_name"'), PosixPath('{},"vgpu"'), PosixPath('"B1.gpu.small"},"cache"'), PosixPath('"/gemini/codeset/rg1oriahgqk1/375876958015197184/latest","data_type_name"'), PosixPath('\\"direct.virtaicloud.com'), PosixPath('"true","mount_path"'), PosixPath('"traindata","space_id"'), PosixPath('{},"instance_ports"'), PosixPath('"storage-docker-registry-agent.gemini-comp'), PosixPath('"latest"},"data_type_name"'), PosixPath('"349896193502875648","version"'), PosixPath('80,"ssh_port"'), PosixPath('3600,"spec_name"'), PosixPath('150,"memory"'), PosixPath('20480,"gpu"'), PosixPath('"/Gemini-Snapshot/traindata/wqubxwo3rzmo/1313/349896193502875648/latest","read_only"'), PosixPath('100,"storage"'), PosixPath('20230915134904","registry"'), PosixPath('"/gemini/code","access_info"'), PosixPath('2,"parameters"'), PosixPath('{}},"data_storage"'), PosixPath('"wqubxwo3rzmo","user_id"'), PosixPath('{"name"'), PosixPath('[{"http_port"'), PosixPath('"Dataset"}],"code_repository"'), PosixPath('"system","password"'), PosixPath('"false","mount_path"'), PosixPath('{"baseUrl"'), PosixPath('22602\\"}","data_path"'), PosixPath('"taskrole1","instances"'), PosixPath('"ROODqJ4XMHH6bCQ8dYiZjg=="},"resource"'), PosixPath('"{\\"cluster_internal_address\\"'), PosixPath('25,"memory"'), PosixPath('8888\\",\\"cluster_external_address\\"'), PosixPath('"/gemini_web/gemini_jupyterlab/~direct.virtaicloud.com~31443~/375877374027239424","port"'), PosixPath('"/gemini/data-1","access_info"'), PosixPath('1,"ratio"'), PosixPath('12288,"memCommitCent"'), PosixPath('6062,"idle_time"')}
  warn(msg)
/root/miniconda3/lib/python3.10/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: WARNING: The following directories listed in your path were found to be non-existent: {PosixPath('8888'), PosixPath('//10.110.200.140'), PosixPath('tcp')}
  warn(msg)
/root/miniconda3/lib/python3.10/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: WARNING: The following directories listed in your path were found to be non-existent: {PosixPath('/gemini/pretrain')}
  warn(msg)
/root/miniconda3/lib/python3.10/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: WARNING: The following directories listed in your path were found to be non-existent: {PosixPath('443'), PosixPath('//10.96.0.1'), PosixPath('tcp')}
  warn(msg)
/root/miniconda3/lib/python3.10/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: WARNING: The following directories listed in your path were found to be non-existent: {PosixPath('用户需知：请提前阅读本文档\\n+----------+-------------------+------------------+-------------------+--------------+--------------------------------+\\n| 存储     | 路径              | 环境变量($+变量) | 权限              | 大小         | 备注                           |\\n+----------+-------------------+------------------+-------------------+--------------+--------------------------------+\\n| 容器储存 | 代码、数据集、    | none             |【开发环境】可读写 | small：20G   | 1.[临时保存]把容器保存成新     |\\n|          | 模型、结果集路径  |                  |【离线训练】可读写 | midium：30G  | 镜像(包含容器存储的数据)       |\\n|          | 以外的所有路径    |                  |                   | large：50G   | 2.容器关闭或重启，会销毁容     |\\n|          |                   |                  |                   | xlarge：100G | 器，容器存储的数据不会保留     |\\n+----------+-------------------+------------------+-------------------+--------------+--------------------------------+\\n| 代码     | /gemini/code      | GEMINI_CODE      |【开发环境】可读写 | 不限制大小   | 1.在项目内挂载，归属于所在的   |\\n|          |                   |                  |【离线训练】只读   |              | 项目                           |\\n|          |                   |                  |                   |              | 2.启动容器后，如果开启了SSH    |\\n|          |                   |                  |                   |              | 或注入了JupyterLab，可以通过   |\\n|          |                   |                  |                   |              | SSH工具或JuupyterLab上传下载   |\\n+----------+-------------------+------------------+-------------------+--------------+--------------------------------+\\n| 数据集   | /gemini/data-1    | GEMINI_DATA_IN1  | 只读 Read Only    | 不限制大小   | 在【数据】栏内上传数据，保存   |\\n|          | /gemini/data-2    | GEMINI_DATA_IN2  |                   |              | 在数据目录下，创建项目时选择   |\\n|          | /gemini/data-3    | GEMINI_DATA_IN3  |                   |              | 会挂载到容器内                 |\\n+----------+-------------------+------------------+-------------------+--------------+--------------------------------+\\n| 模型     | /gemini/pretrain  | GEMINI_PRETRAIN  | 只读 Read Only    | 不限制大小   | none                           |\\n|          | /gemini/pretrain2 | GEMINI_PRETRAIN2 |                   |              |                                |\\n|          | /gemini/pretrain3 | GEMINI_PRETRAIN3 |                   |              |                                |\\n+----------+-------------------+------------------+-------------------+--------------+--------------------------------+\\n| 结果集   | /gemini/output    | GEMINI_DATA_OUT  | 仅【离线训练】    | 不限制大小   | 挂载在项目内，归属于所在的项目 |\\n|          |                   |                  | 有此功能可读写    |              |                                |\\n+----------+-------------------+------------------+-------------------+--------------+--------------------------------+')}
  warn(msg)
/root/miniconda3/lib/python3.10/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: WARNING: The following directories listed in your path were found to be non-existent: {PosixPath('/gemini/output')}
  warn(msg)
/root/miniconda3/lib/python3.10/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: WARNING: The following directories listed in your path were found to be non-existent: {PosixPath('/gemini/data-3')}
  warn(msg)
/root/miniconda3/lib/python3.10/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: WARNING: The following directories listed in your path were found to be non-existent: {PosixPath('/gemini/data-2')}
  warn(msg)
/root/miniconda3/lib/python3.10/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: WARNING: The following directories listed in your path were found to be non-existent: {PosixPath('/gemini/predata')}
  warn(msg)
/root/miniconda3/lib/python3.10/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: WARNING: The following directories listed in your path were found to be non-existent: {PosixPath('/gemini/pretrain2')}
  warn(msg)
/root/miniconda3/lib/python3.10/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: WARNING: The following directories listed in your path were found to be non-existent: {PosixPath('/gemini/pretrain3')}
  warn(msg)
CUDA_SETUP: WARNING! libcudart.so not found in any environmental path. Searching in backup paths...
/root/miniconda3/lib/python3.10/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: Found duplicate ['libcudart.so', 'libcudart.so.11.0', 'libcudart.so.12.0'] files: {PosixPath('/usr/local/cuda/lib64/libcudart.so.11.0'), PosixPath('/usr/local/cuda/lib64/libcudart.so')}.. We'll flip a coin and try one of these, in order to fail forward.
Either way, this might cause trouble in the future:
If you get `CUDA error: invalid device function` errors, the above might be the cause and the solution is to make sure only one ['libcudart.so', 'libcudart.so.11.0', 'libcudart.so.12.0'] in the paths that we search based on your env.
  warn(msg)
CUDA SETUP: CUDA runtime path found: /usr/local/cuda/lib64/libcudart.so.11.0
CUDA SETUP: Highest compute capability among GPUs detected: 8.6
CUDA SETUP: Detected CUDA version 118
CUDA SETUP: Loading binary /root/miniconda3/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cuda118.so...
[2023-11-04 05:35:36,401] [INFO] [real_accelerator.py:158:get_accelerator] Setting ds_accelerator to cuda (auto detect)
2023-11-04 05:35:47.257156: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2023-11-04 05:35:49.152841: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-11-04 05:35:50.391047: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Loading weights [6ce0161689] from /gemini/code/stable-diffusion-webui/models/Stable-diffusion/v1-5-pruned-emaonly.safetensors
Creating model from config: /gemini/code/stable-diffusion-webui/configs/v1-inference.yaml
Running on local URL:  http://0.0.0.0:7860

To create a public link, set `share=True` in `launch()`.
Startup time: 72.1s (prepare environment: 14.8s, import torch: 30.7s, import gradio: 1.7s, setup paths: 4.6s, import ldm: 0.3s, initialize shared: 5.3s, other imports: 6.4s, setup codeformer: 1.5s, setup gfpgan: 0.1s, load scripts: 2.8s, load upscalers: 0.2s, initialize extra networks: 0.8s, create ui: 1.6s, gradio launch: 1.1s).
Applying attention optimization: xformers... done.
Model loaded in 104.4s (load weights from disk: 3.0s, create model: 0.8s, apply weights to model: 97.4s, apply half(): 0.4s, load textual inversion embeddings: 0.3s, calculate empty prompt: 2.4s).
```

体验
``` prompt
(masterpiece),(best quality),(Realistic photos),a cute girl, war a cap, headphones, summer, sunset, mountain road, flowers, nice weather, healing sense, detailed, half-length shot, anime style, 8k
```
![](img/Stable-diffusion/3-txt2img-demo.png)
