# 🎨 DDColor
[![arXiv](https://img.shields.io/badge/arXiv-2212.11613-b31b1b.svg)](https://arxiv.org/abs/2212.11613)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FF8000)](https://huggingface.co/piddnad/DDColor-models)
[![ModelScope 演示](https://img.shields.io/badge/%F0%9F%91%BE%20ModelScope-Demo-8A2BE2)](https://www.modelscope.cn/models/damo/cv_ddcolor_image-colorization/summary)
[![Replicate](https://replicate.com/piddnad/ddcolor/badge)](https://replicate.com/piddnad/ddcolor)
![visitors](https://visitor-badge.laobi.icu/badge?page_id=piddnad/DDColor)

ICCV 2023 论文 "DDColor: Towards Photo-Realistic Image Colorization via Dual Decoders" 的官方 PyTorch 实现。

> 康晓阳、杨涛、欧阳文琦、任培然、李令芝、谢宣松
> *达摩院，阿里巴巴集团*

🪄 DDColor 能够为历史黑白老照片提供生动自然的上色效果。

<p align="center">
  <img src="assets/teaser.webp" width="100%">
</p>

🎲 它甚至可以为动漫游戏中的风景进行上色/重新着色，将动画场景转化为逼真的现实风格！（图片来源：原神）

<p align="center">
  <img src="assets/anime_landscapes.webp" width="100%">
</p>


## 最新动态
- [2024-01-28] 支持通过 🤗 Hugging Face 进行推理！感谢 @[Niels](https://github.com/NielsRogge) 的建议和示例代码以及 @[Skwara](https://github.com/Skwarson96) 修复的 bug。
- [2024-01-18] 新增 Replicate 演示和 API！感谢 @[Chenxi](https://github.com/chenxwh)。
- [2023-12-13] 发布 DDColor-tiny 预训练模型！
- [2023-09-07] 新增模型库并发布三个预训练模型！
- [2023-05-15] 训练和推理代码发布！
- [2023-05-05] 在线演示上线！


## 在线演示
在 [ModelScope](https://www.modelscope.cn/models/damo/cv_ddcolor_image-colorization/summary) 和 [Replicate](https://replicate.com/piddnad/ddcolor) 上体验我们的在线演示。


## 方法简介
*简述：* DDColor 利用多尺度视觉特征优化**可学习颜色令牌**（即颜色查询），在自动图像上色任务上达到了最先进的性能。

<p align="center">
  <img src="assets/network_arch.jpg" width="100%">
</p>


## 安装
### 环境要求
- Python >= 3.7
- PyTorch >= 1.7

### 使用 conda 安装（推荐）

```sh
conda create -n ddcolor python=3.9
conda activate ddcolor
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

# 如需训练，请安装以下额外依赖和 basicsr
pip install -r requirements.train.txt
python3 setup.py develop
```

## 快速开始
### 使用本地脚本推理（无需 `basicsr`）
1. 下载预训练模型：

```python
from modelscope.hub.snapshot_download import snapshot_download

model_dir = snapshot_download('damo/cv_ddcolor_image-colorization', cache_dir='./modelscope')
print('模型资源已保存至 %s' % model_dir)
```

2. 运行推理：

```sh
python scripts/infer.py --model_path ./modelscope/damo/cv_ddcolor_image-colorization/pytorch_model.pt --input ./assets/test_images
```
或者
```sh
sh scripts/inference.sh
```

### 使用 Hugging Face 推理
通过 Hugging Face Hub 加载模型：

```python
from huggingface_hub import PyTorchModelHubMixin
from ddcolor import DDColor

class DDColorHF(DDColor, PyTorchModelHubMixin):
    def __init__(self, config=None, **kwargs):
        if isinstance(config, dict):
            kwargs = {**config, **kwargs}
        super().__init__(**kwargs)

ddcolor_paper_tiny = DDColorHF.from_pretrained("piddnad/ddcolor_paper_tiny")
ddcolor_paper      = DDColorHF.from_pretrained("piddnad/ddcolor_paper")
ddcolor_modelscope = DDColorHF.from_pretrained("piddnad/ddcolor_modelscope")
ddcolor_artistic   = DDColorHF.from_pretrained("piddnad/ddcolor_artistic")
```

或直接运行以下命令进行模型推理：

```sh
python scripts/infer.py --model_name ddcolor_modelscope --input ./assets/test_images
# model_name 可选: [ddcolor_paper | ddcolor_modelscope | ddcolor_artistic | ddcolor_paper_tiny]
```

### 使用 ModelScope 推理
1. 安装 modelscope：

```sh
pip install modelscope
```

2. 运行推理：

```python
import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

img_colorization = pipeline(Tasks.image_colorization, model='damo/cv_ddcolor_image-colorization')
result = img_colorization('https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/audrey_hepburn.jpg')
cv2.imwrite('result.png', result[OutputKeys.OUTPUT_IMG])
```

以上代码会自动下载 `ddcolor_modelscope` 模型（见[模型库](#模型库)）并执行推理。模型文件 `pytorch_model.pt` 可在本地路径 `~/.cache/modelscope/hub/damo` 中找到。

### Gradio 演示
安装 gradio 及其他必要的库：

```sh
pip install gradio gradio_imageslider
```

然后运行以下命令启动演示：

```sh
python demo/gradio_app.py
```

## 模型库
我们提供了多个不同版本的预训练模型，详情请查看 [模型库](MODEL_ZOO.md)。

| 模型 | HuggingFace 链接 | 描述 | 备注 |
| --- | :--- | :--- | :--- |
| ddcolor_paper | [链接](https://huggingface.co/piddnad/ddcolor_paper) | 在 ImageNet 上训练的 DDColor-L | 论文模型，仅在需要复现论文中部分图片时使用。 |
| ddcolor_modelscope（***默认***） | [链接](https://huggingface.co/piddnad/ddcolor_modelscope) | 在 ImageNet 上训练的 DDColor-L | 我们使用了与 [BigColor](https://github.com/KIMGEONUNG/BigColor/issues/2#issuecomment-1196287574) 相同的数据清洗方案训练此模型，可在几乎不降低 FID 性能的情况下获得最佳定性结果。如需测试 ImageNet 以外的图片，建议默认使用此模型。也可通过 ModelScope 轻松下载。 |
| ddcolor_artistic | [链接](https://huggingface.co/piddnad/ddcolor_artistic) | 在 ImageNet + 私有数据上训练的 DDColor-L | 我们使用包含大量高质量艺术图片的扩展数据集训练此模型。此外，训练过程中未使用色彩度损失，因此可能出现更少的不合理颜色伪影。如需尝试不同的上色效果，可使用此模型。 |
| ddcolor_paper_tiny | [链接](https://huggingface.co/piddnad/ddcolor_paper_tiny) | 在 ImageNet 上训练的 DDColor-T | DDColor 最轻量级版本，使用与 ddcolor_paper 相同的训练方案。 |


## 训练
1. 数据集准备：下载 [ImageNet](https://www.image-net.org/) 数据集或创建自定义数据集。使用以下脚本获取数据集列表文件：

```sh
python scripts/get_meta_file.py
```

2. 下载 [ConvNeXt](https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth) 和 [InceptionV3](https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth) 的预训练权重，并将它们放置在 `pretrain` 文件夹中。

3. 在 `options/train/train_ddcolor.yml` 中指定 `meta_info_file` 及其他选项。

4. 开始训练：

```sh
sh scripts/train.sh
```

## ONNX 导出
支持 ONNX 模型导出。

1. 安装依赖：

```sh
pip install onnx==1.16.1 onnxruntime==1.19.2 onnxsim==0.4.36
```

2. 使用示例：

```sh
python scripts/export_onnx.py --model_path pretrain/ddcolor_paper_tiny.pth --export_path weights/ddcolor-tiny.onnx
```

使用 `ddcolor_paper_tiny` 模型的 ONNX 导出演示请参阅[此处](demo/colorization_pipeline_onnxruntime.ipynb)。


## 引用

如果我们的工作对您的研究有所帮助，请考虑引用：

```
@inproceedings{kang2023ddcolor,
  title={DDColor: Towards Photo-Realistic Image Colorization via Dual Decoders},
  author={Kang, Xiaoyang and Yang, Tao and Ouyang, Wenqi and Ren, Peiran and Li, Lingzhi and Xie, Xuansong},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={328--338},
  year={2023}
}
```

## 致谢
感谢 BasicSR 的作者提供了优秀的训练流程框架。

> Xintao Wang, Ke Yu, Kelvin C.K. Chan, Chao Dong and Chen Change Loy. BasicSR: Open Source Image and Video Restoration Toolbox. https://github.com/xinntao/BasicSR, 2020.

部分代码改编自 [ColorFormer](https://github.com/jixiaozhong/ColorFormer)、[BigColor](https://github.com/KIMGEONUNG/BigColor)、[ConvNeXt](https://github.com/facebookresearch/ConvNeXt)、[Mask2Former](https://github.com/facebookresearch/Mask2Former) 和 [DETR](https://github.com/facebookresearch/detr)。感谢他们的杰出工作！