A vision-language-model (VLM) inference engine. Particularly suitable for running on edge computing platforms(Nvidia Jetson AGX Thor/Orin).

## Supportted Models

- [Qwen3-VL](https://huggingface.co/collections/Qwen/qwen3-vl)
- [Qwen2.5-VL](https://huggingface.co/collections/Qwen/qwen25-vl)
- [Xiaomi-MiMo-VL-Miloco](https://huggingface.co/xiaomi-open-source/Xiaomi-MiMo-VL-Miloco-7B)

For our test, the performance is comparable to mainstream frameworks(vLLM or SGLang). 
 - Jetson AGX orin (batch=1, input-image:800x477) Decode throughput:
    - [Qwen3-VL-8B](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) 11tok/s(w16a16), 30tok/s(w4a16)
    - [Qwen3-VL-2B](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) 47tok/s(w16a16), 82tok/s(w4a16)
 
 - nvidia-H20 (batch=1, input-image:2048x1365) Decode throughput:
    - [Qwen3-VL-8B](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) ~160 tok/s(w16a16), ~180 tok/s(w4a16)
    - [Qwen3-VL-2B](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) ~330 tok/s(w16a16), ~400 tok/s(w4a16)
    - [Xiaomi-Miloco-7B](https://huggingface.co/xiaomi-open-source/Xiaomi-MiMo-VL-Miloco-7B) ~180 tok/s(w16a16)

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

 - **Quantization:**

  - Support w4a16 awq quantization, you can use [awq-model](https://huggingface.co/cpatonn/Qwen3-VL-8B-Instruct-AWQ-4bit) directly or use online quantization. 

  - If you use online quantization, please add `-q ./online_quantization_marlin.json`

### server-client-mode

此模式需要先启动一个api server，然后通过发送请求运行模型生成

>launch server

`purevlm-serve -m /path/to/qwen3-vl --host 0.0.0.0 --port 8002 --use-cuda-graph`

>launch client requests

`cd examples`

`python client_test.py --server http://localhost:8002 --image ./demo.jpeg --message "描述这张图片"`

## Acknowledgement

We learned the design and reused code from the following projects: [transformers](https://github.com/huggingface/transformers), [vLLM](https://github.com/vllm-project/vllm), [Flash-attention](https://github.com/Dao-AILab/flash-attention)