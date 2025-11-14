A vision-language-model (VLM) inference engine. Particularly suitable for running on edge computing platforms(Nvidia Jetson AGX Thor/Orin).

For our test, the performance is comparable to mainstream frameworks(vLLM or SGLang). 
 - Qwen3-VL-8B: decode throughput(batch=1, in nvidia-H20): ~160 tok/s(w16a16), ~180 tok/s(w4a16)

## Install

### Requirements

**Requirements:**

- CUDA toolkit
- PyTorch 2.7 and above.
- Transformers

**To Install:**

editable:

`pip install -v -e . --no-build-isolation --no-deps`

pacakge:

`python setup.py install`

**Optional:**

If you wish to use cuda-graph to get more performance improve, Please install [Flash-Attention](https://github.com/Dao-AILab/flash-attention).

## Usage

支持offline运行与server-client模式运行两种模式

### offline-mode

`cd examples`

`python offline_demo.py -m /path/to/Qwen3-VL-8B-Instruct -p '描述这张图片' -t 0.0 -im ./demo.jpeg --use-cuda-graph`

If use cuda graph, Please keep flash-attention installed.

To use online quantization, please add `-q /path/to/quant/json`

### server-client-mode

此模式需要先启动一个api server，然后通过发送请求运行模型生成

>launch server

`purevlm-serve -m /path/to/qwen3-vl --host 0.0.0.0 --port 8002 --use-cuda-graph`

>launch client requests

`cd examples`

`python client_test.py --server http://localhost:8002 --image ./demo.jpeg --message "描述这张图片"`

## Acknowledgement

We learned the design and reused code from the following projects: [transformers](https://github.com/huggingface/transformers), [vLLM](https://github.com/vllm-project/vllm), [Flash-attention](https://github.com/Dao-AILab/flash-attention)