
<p align="left">
  <img src=assets/logo.svg alt="logo" width="70%" />
</p>

<a href="#">
  <img src="https://img.shields.io/badge/python-3.6-f0f0f0"/>
</a>

<a href="#">
  <img src="https://img.shields.io/badge/pytorch-1.6.0-f0f0f0"/>
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

<a href= "https://twitter.com/intent/tweet?text=iSegmentation:%20Semantic%20Segmentation%20Zoo%20Built%20by%20PyTorch.&url=https://github.com/TaikiInoue/iSegmentation">
        <img src="https://img.shields.io/twitter/url/https/shields.io.svg?style=social" />
</a>

<br>
<br>

iSegmentation is a semantic segmentation zoo built by PyTorch. This repository is still a work in progress, so README.md may contain incorrect information.

![demo image](assets/demo.png)

<br>

## 1. Key Features

### ■ Define Models with YAML File
![Screen Shot 2020-10-06 at 19 33 25](https://user-images.githubusercontent.com/29189728/95190833-d586ab80-080a-11eb-8553-ce8969526bf4.png)

<br>

### ■ Provide Reliable Models

<img width="788" alt="Screen Shot 2020-10-06 at 19 41 05" src="https://user-images.githubusercontent.com/29189728/95191608-eedc2780-080b-11eb-984a-f31155bb64bd.png">

<br>

### ■ Easy to Call Models

```
from iseg.models import DeepLabV3Plus, FastFCN, UNet

deeplabv3plus = DeepLabV3Plus()
fastfcn = FastFCN()
unet = UNet()
```

<br>

## 2. Requirements

- Tensor Core Supported GPU
- CUDA 10.2
- Nvidia-Docker2
- [iClassification: Classification Zoo Built by PyTorch](https://github.com/TaikiInoue/iClassification)

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

| Model         | Backbone | Crop Size | Training Time (h) | mIoU | Config File |
| :-            | :-       | :-        | :-                | :-   | :-          |
| DeepLabV3Plus | ResNet50 | -         | -                 | -    | -           |
| FastFCN       | ResNet50 | -         | -                 | -    | -           |
| PSPNet        | ResNet50 | -         | -                 | -    | -           |
| UNet          | ResNet50 | -         | -                 | -    | -           |


<br>

## 4. Roadmaps

- Implement UNet
- Implement DeepLabV3
- Implement FastFCN
- Support pretrained backbones
- Create benchmark table on ADE20K dataset
