# RF-DETR Model Adapter

## Introduction

This repository provides an integration between the [RF-DETR (Region-Free Detection Transformer)](https://github.com/IDEA-Research/RFDet) and the [Dataloop](https://dataloop.ai/) platform.

RF-DETR is a real-time, transformer-based object detection model that delivers both high accuracy and efficient inference. It is the first real-time model to surpass 60 AP on the Microsoft COCO benchmark while maintaining competitive speed and performance at base sizes. Additionally, it sets a new standard on the RF100-VL benchmark, which evaluates model generalization to real-world domains.

Thanks to its compact architecture, RF-DETR is well-suited for edge deployments where low latency and high precision are essential. This adapter connects RF-DETR to the Dataloop ecosystem, enabling streamlined training, evaluation, and deployment workflows for object detection tasks at scale.

## Why RF-DETR?

Recent changes in the licensing of YOLO models, particularly YOLOv8, have introduced restrictions that impact their use in commercial applications. YOLOv8 is now licensed under AGPL-3.0, which requires that any derivative works or applications that use the software and are distributed must also be open-sourced under the same license. For organizations that wish to keep their source code proprietary, this necessitates obtaining a commercial Enterprise License from Ultralytics.

In contrast, RF-DETR is released under the permissive Apache 2.0 license, allowing for free use in both open-source and proprietary projects without the need for additional licensing. RF-DETR offers a license-friendly alternative for commercial and proprietary applications.

## Requirements

- dtlpy  
- dtlpy-converters  
- torch  
- torchvision  
- numpy  
- An account on the [Dataloop platform](https://console.dataloop.ai/)

## Installation

To use this adapter, make sure you have a [project](https://developers.dataloop.ai/tutorials/getting_started/sdk_overview/chapter/#to-create-a-new-project) and a [dataset](https://developers.dataloop.ai/tutorials/data_management/datasets_and_versioning/chapter) in your Dataloop account.

For training purposes, ensure your dataset includes subsets (e.g., "train", "validation").

## Training and Fine-Tuning

To fine-tune RF-DETR on a custom dataset using the SDK, follow [this tutorial](https://developers.dataloop.ai/tutorials/model_management).

## Supported Model

- **RF-DETR**: This adapter supports multiple pretrained variants of RF-DETR. By default, the model uses `rf-detr-base-coco.pth`. You can change the weights by updating the `weights_filename` field in the model configuration and uploading the selected file as an artifact.

### Available Weights:

- `rf-detr-base-coco.pth` – Balanced for real-time performance and accuracy. Recommended for most use cases, especially inference on edge devices.
- `rf-detr-base-2.pth` – Similar to `base-coco` but optimized for further fine-tuning on custom datasets.
- `rf-detr-large.pth` – Larger variant with improved accuracy potential for fine-tuning. Slower inference, not suitable for real-time edge applications.


## Configuration

For information on how to configure training parameters (e.g., learning rate, batch size, number of epochs), please refer to the official [RF-DETR training section](https://github.com/roboflow/rf-detr/blob/develop/README.md#training).
