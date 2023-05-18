## 构建小样本目标检测模块

**该模块的设计基于[DiffusionDet](https://github.com/ShoufaChen/DiffusionDet)** ，感谢其项目作者ShoufaChen的支持。



## 环境配置

- Linux with Python ≥ 3.7.
- PyTorch ≥ 1.7.0 以及相匹配的[torchvision](https://github.com/pytorch/vision/).
- OpenCV.
- Detectron2 v0.6.
- tqdm.
- 使您的Linux机支持screen命令

**注意：**请使用conda 或mini conda管理python环境，且用于该项目的环境需命名为alice

或者，您可以仔细阅读该模块的[上级项目](https://github.com/sixone-Jiang/LabelOne2All)以修改bin/tools/RemoteSSH.py文件中 conda activate alice 为 您设定的环境名。



## 数据集

请将您的无标签数据集图像上传到**datasets/myVocdata/JPEGImages/**下



## 预训练模型下载

您可以下载full coco 80 swin_model   [Download](https://github.com/ShoufaChen/DiffusionDet/releases/download/v0.1/diffdet_coco_swinbase.pth)

并将其放在models/swin_run.pth作为您的基础模型

后续，我们将上传更多的模型支持



## 测试该模块是否正常运行

```shell
python demo.py --config-file configs/diffdet.coco.swinbase.yaml \
    --input image.jpg --output output.jpg --opts MODEL.WEIGHTS models/swin_run.pth
```



## License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](https://github.com/ShoufaChen/DiffusionDet/blob/main/LICENSE) for details.