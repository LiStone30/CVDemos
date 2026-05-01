



# 不进入 ubuntu 的 容器，在 ubuntu上进行开发
## 构建一个开发镜像
python:3.12-slim-bookworm

podman pull docker.1ms.run/library/python:3.12-slim-bookworm

podman build -f dockerfile_dev -t python_dev:312 .

localhost/python_dev:312
## 运行临时开发容器
podman run -it --rm localhost/python_dev:312  bash

## 测试下 podman 容器的GPU支持
podman pull docker.1ms.run/nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04

podman run --rm -it --device nvidia.com/gpu=all docker.1ms.run/nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04 nvidia-smi


# 激活环境
source .venv/bin/activate

# 测试下网络
curl -i https://www.google.com


curl -x http://127.0.0.1:7897 https://www.google.com