
<p align="left">
  <img src=assets/logo.svg alt="logo" width="70%" />
</p>

<a href="#">
  <img src="https://img.shields.io/badge/python-3.6-f0f0f0"/>
</a>

<a href="#">
  <img src="https://img.shields.io/badge/cuda-10.2-f0f0f0"/>
</a>

<a href="#">
    <img src="https://img.shields.io/badge/code style- black | flake8 | isort -f0f0f0" />
</a>

<a href="#">
  <img src="https://img.shields.io/badge/lisence-MIT-f0f0f0" />
</a>

<br>
<br>

iSegmentation is a semantic segmentation zoo built by PyTorch. This repository is still a work in progress, so README.md may contain incorrect information.

![demo image](assets/demo.png)

<br>

## 1. Key Features

### 1.1. Define Models with YAML File

### 1.2. Provide Reliable Models

### 1.3. Easy to Call Models

<br>

## 2. Requirements

- Tensor Core Supported GPU
- CUDA 10.2
- Nvidia-Docker2

<br>

## 3. Getting Started

Download this repository from GitHub
```
git clone https://github.com/TaikiInoue/iSegmentation.git
cd iSegmentation
```

Build docker image
```
make docker_build
```

Start docker container
```
make docker_run
```

Run a semantic segmentation model. Currently, `UNet`, `DeepLabV3Plus` and `FastFCN` are available in `[model name]`
```
make run_[model name]
```

<br>

## 3. Benchmarks

| Model | Backbone | Crop Size | Training Time (h) | mIoU | Config File |
| --    | --       | --        | --                | --   | --          |

<br>

## 4. Roadmaps

- Implement UNet
- Implement DeepLabV3
- Implement FastFCN
- Support pretrained backbones
