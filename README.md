A pure VLM model inference engine.

## Install

`pip install -v -e . -i https://pypi.tuna.tsinghua.edu.cn/simple --no-build-isolation --no-deps`

## Usage

支持offline运行与server-client模式运行两种模式

### offline-mode

`cd examples`

`python offline_demo.py -m /path/to/Qwen3-VL -p '帮我找下音箱' -t 0.0 -im ./jiaju-demo.png`

To use online quantization, please add `-q /path/to/quant/json`

### server-client-mode

此模式需要先启动一个api server，然后通过发送请求运行模型生成

>launch server

`purevlm-serve -m /path/to/qwen3-vl --host 0.0.0.0 --port 8002`

>launch client requests

`cd examples`

`python client_test.py --server http://localhost:8002 --image ./jiaju-demo.png --message "帮我找下音箱"`