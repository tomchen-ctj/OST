<div align="center">
<h1> 【CVPR'24】OST: Refining Text Knowledge with Optimal Spatio-Temporal Descriptor for General Video Recognition </h1>

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ost-refining-text-knowledge-with-optimal/zero-shot-action-recognition-on-kinetics)](https://paperswithcode.com/sota/zero-shot-action-recognition-on-kinetics?p=ost-refining-text-knowledge-with-optimal)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ost-refining-text-knowledge-with-optimal/zero-shot-action-recognition-on-hmdb51)](https://paperswithcode.com/sota/zero-shot-action-recognition-on-hmdb51?p=ost-refining-text-knowledge-with-optimal)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ost-refining-text-knowledge-with-optimal/zero-shot-action-recognition-on-ucf101)](https://paperswithcode.com/sota/zero-shot-action-recognition-on-ucf101?p=ost-refining-text-knowledge-with-optimal)

[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://tomchen-ctj.github.io/OST/)
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2312.00096.pdf)

[Tongjia Chen](https://tomchen-ctj.github.io/)<sup>1</sup>, [Hongshan Yu]()<sup>1</sup>, [Zhengeng Yang](https://scholar.google.com/citations?user=26tsC5kAAAAJ&hl=en&oi=ao)<sup>2</sup>, [Zechuan Li](https://scholar.google.com/citations?user=b0GewmQAAAAJ&hl=en&oi=ao)<sup>1</sup>, [Wei Sun]()<sup>1</sup>, [Chen Chen](https://www.crcv.ucf.edu/chenchen/)<sup>3</sup>

<sup>1</sup>[HNU](https://www-en.hnu.edu.cn/), <sup>2</sup>[HNNU](https://english.hunnu.edu.cn/), <sup>3</sup>[CRCV, UCF](http://crcv.ucf.edu/)
</div>

In this work, we introduce a novel general video recognition pipeline **OST**. We prompt an LLM to augment category names into *Spatio-Temporal Descriptors* and refine the semantic knowledge via *Optimal Descriptor Solver*.

<div align=center>
<img width="500" alt="image" src="imgs/teaser.png">
</div>
Dominant pipelines propose to tackle the visual discrepancies with additional temporal learners while overlooking the textual discrepancy between descriptive narratives and concise category names. This oversight results in a less separable latent space, which may hinder video recognition.






<div align=center>
<img width="1080" alt="image" src="imgs/pipeline.png">
</div>

We query the Large Language Model to augment category names to generate corresponding *Category Descriptors*. The descriptors disentangled category names into *Spatio-Temporal Descriptors* for static visual cues and temporal evolution, respectively. To fully refine the textual knowledge, we propose *Optimal Descriptor Solver* that adaptively aligns descriptors with video frames. An optimal matching flow is calculated through the iterative solving of the entropy-regularized OT problem to assign optimal descriptors for each video instance.

## Todo

- [x] Code Release
- [x] Environment
- [x] Checkpoint Release
- [ ] Fully-supervised code & model weights

## Environments

Our codebase is mainly built on [ViFi-CLIP](https://github.com/muzairkhattak/ViFi-CLIP), please follow the [instruction]( https://github.com/muzairkhattak/ViFi-CLIP/blob/main/docs/INSTALL.md) provided in their repository to build environments. 

(*Note that you may need to build your environments on mmcv 1.x*)

## Train & Eval

Fow all the experiments illustrated in the main paper, we provide config files in the `configs` folder. For example, to train OST on Kinetics-400, you can run the following command:

```shell
python -m torch.distributed.launch --nproc_per_node=8 \ 
main_nce.py -cfg configs/zero_shot/train/k400/16_32_ost.yaml --output /PATH/TO/OUTPUT 
```



To evaluate a model, please use the specific config file in the `configs` folder according to the dataset and data splits. To evaluate OST in the zero-shot setting with 32 frames on UCF-101 zero-shot split-1, you can run the command below:

```shell
python -m torch.distributed.launch --nproc_per_node=8 \
main_nce.py -cfg configs/zero_shot/eval/ucf/16_32_ost_zs_ucf101_split1.yaml --output /PATH/TO/OUTPUT \
--only_test --resume /PATH/TO/CKPT
```

Please note that we use 8 GPUs in all of our main experiments, and the results may vary due to different environment settings and hardwares.

## Checkpoints

We use OpenAI pretrained CLIP-B/16 model in all of our experiments. We provide checkpoints of our OST in zero-shot and few-shot settings below. All of the model checkpoints are available in the [HuggingFace Space](https://huggingface.co/spaces/tomchen-ctj/OST/tree/main) 🤗.

**Zero-shot setting**

For the zero-shot setting, the model is first fine-tuned on Kinetics-400 and then directly evaluated on 3 downstream datasets. So here we provide our Kinetics-400 fine-tuned model weights for reproducing the zero-shot results illustrated in our main paper.

| Config                                             | Input      | HMDB-51 | UCF-101 | Kinetics-600 | Checkpoints                                                  |
| -------------------------------------------------- | ---------- | ------- | ------- | ------------ | ------------------------------------------------------------ |
| [OST](configs/zero_shot/train/k400/16_32_ost.yaml) | $32*224^2$ | 55.9    | 79.7    | 75.1         | [Link](https://huggingface.co/spaces/tomchen-ctj/OST/blob/main/checkpoints/zero_shot/OST_10_epochs_k400_finetuned.pth) |



**Few-shot setting**

For few-shot setting, we follows the evaluation protocal of mainstream pipelines. We evaluate OST in two different settings (Directly tuning on CLIP & Fine-tuned on K400).

*Directly tuning on CLIP*

| Config                                                       | Input      | Shots | Dataset                | Top-1 Acc. | Checkpoints                                                  |
| ------------------------------------------------------------ | ---------- | ----- | ---------------------- | ---------- | ------------------------------------------------------------ |
| [OST](configs/few_shot/finetuning_few_shot/hmdb51/16_32_ost_2_shot.yaml) | $32*224^2$ | 2     | HMDB-51                | 59.1       | [Link](https://huggingface.co/spaces/tomchen-ctj/OST/blob/main/checkpoints/few_shot/hmdb/k2_few_shot_hmdb_OST.pth) |
| [OST](configs/few_shot/finetuning_few_shot/hmdb51/16_32_ost_4_shot.yaml) | $32*224^2$ | 4     | HMDB-51                | 62.9       | [Link](https://huggingface.co/spaces/tomchen-ctj/OST/blob/main/checkpoints/few_shot/hmdb/k4_few_shot_hmdb_OST.pth) |
| [OST](configs/few_shot/finetuning_few_shot/hmdb51/16_32_ost_8_shot.yaml) | $32*224^2$ | 8     | HMDB-51                | 64.9       | [Link](https://huggingface.co/spaces/tomchen-ctj/OST/blob/main/checkpoints/few_shot/hmdb/k8_few_shot_hmdb_OST.pth) |
| [OST](configs/few_shot/finetuning_few_shot/hmdb51/16_32_ost_16_shot.yaml) | $32*224^2$ | 16    | HMDB-51                | 68.2       | [Link](https://huggingface.co/spaces/tomchen-ctj/OST/blob/main/checkpoints/few_shot/hmdb/k16_few_shot_hmdb_OST.pth) |
| [OST](configs/few_shot/finetuning_few_shot/ucf101/16_32_ost_2_shot.yaml) | $32*224^2$ | 2     | UCF-101                | 82.5       | [Link](https://huggingface.co/spaces/tomchen-ctj/OST/blob/main/checkpoints/few_shot/ucf/k2_few_shot_ucf_OST.pth) |
| [OST](configs/few_shot/finetuning_few_shot/ucf101/16_32_ost_4_shot.yaml) | $32*224^2$ | 4     | UCF-101                | 87.5       | [Link](https://huggingface.co/spaces/tomchen-ctj/OST/blob/main/checkpoints/few_shot/ucf/k4_few_shot_ucf_OST.pth) |
| [OST](configs/few_shot/finetuning_few_shot/ucf101/16_32_ost_8_shot.yaml) | $32*224^2$ | 8     | UCF-101                | 91.7       | [Link](https://huggingface.co/spaces/tomchen-ctj/OST/blob/main/checkpoints/few_shot/ucf/k8_few_shot_ucf_OST.pth) |
| [OST](configs/few_shot/finetuning_few_shot/ucf101/16_32_ost_16_shot.yaml) | $32*224^2$ | 16    | UCF-101                | 93.9       | [Link](https://huggingface.co/spaces/tomchen-ctj/OST/blob/main/checkpoints/few_shot/ucf/k16_few_shot_ucf_OST.pth) |
| [OST](configs/few_shot/finetuning_few_shot/ssv2/16_32_ost_2_shot.yaml) | $32*224^2$ | 2     | Something-Something V2 | 7.0        | [Link](https://huggingface.co/spaces/tomchen-ctj/OST/blob/main/checkpoints/few_shot/ssv2/k2_few_shot_ssv2_OST.pth) |
| [OST](configs/few_shot/finetuning_few_shot/ssv2/16_32_ost_4_shot.yaml) | $32*224^2$ | 4     | Something-Something V2 | 7.7        | [Link](https://huggingface.co/spaces/tomchen-ctj/OST/blob/main/checkpoints/few_shot/ssv2/k4_few_shot_ssv2_OST.pth) |
| [OST](configs/few_shot/finetuning_few_shot/ssv2/16_32_ost_8_shot.yaml) | $32*224^2$ | 8     | Something-Something V2 | 8.9        | [Link](https://huggingface.co/spaces/tomchen-ctj/OST/blob/main/checkpoints/few_shot/ssv2/k8_few_shot_ssv2_OST.pth) |
| [OST](configs/few_shot/finetuning_few_shot/ssv2/16_32_ost_16_shot.yaml) | $32*224^2$ | 16    | Something-Something V2 | 12.2       | [Link](https://huggingface.co/spaces/tomchen-ctj/OST/blob/main/checkpoints/few_shot/ssv2/k16_few_shot_ssv2_OST.pth) |

*Fine-tuned on K400*

Please note for this setting, you only need to replace the original CLIP with our Kinetics-400 finetuned model.

| Config                                                       | Input      | Shots | Dataset                | Top-1 Acc. | Checkpoints                                                  |
| ------------------------------------------------------------ | ---------- | ----- | ---------------------- | ---------- | ------------------------------------------------------------ |
| [OST](configs/few_shot/finetuning_few_shot/hmdb51/16_32_ost_2_shot.yaml) | $32*224^2$ | 2     | HMDB-51                | 64.8       | [Link](https://huggingface.co/spaces/tomchen-ctj/OST/blob/main/checkpoints/few_shot/hmdb/k2_few_shot_hmdb_OST_k400_ft.pth) |
| [OST](configs/few_shot/finetuning_few_shot/hmdb51/16_32_ost_4_shot.yaml) | $32*224^2$ | 4     | HMDB-51                | 66.7       | [Link](https://huggingface.co/spaces/tomchen-ctj/OST/blob/main/checkpoints/few_shot/hmdb/k4_few_shot_hmdb_OST_k400_ft.pth) |
| [OST](configs/few_shot/finetuning_few_shot/hmdb51/16_32_ost_8_shot.yaml) | $32*224^2$ | 8     | HMDB-51                | 69.2       | [Link](https://huggingface.co/spaces/tomchen-ctj/OST/blob/main/checkpoints/few_shot/hmdb/k8_few_shot_hmdb_OST_k400_ft.pth) |
| [OST](configs/few_shot/finetuning_few_shot/hmdb51/16_32_ost_16_shot.yaml) | $32*224^2$ | 16    | HMDB-51                | 71.6       | [Link](https://huggingface.co/spaces/tomchen-ctj/OST/blob/main/checkpoints/few_shot/hmdb/k16_few_shot_hmdb_OST_k400_ft.pth) |
| [OST](configs/few_shot/finetuning_few_shot/ucf101/16_32_ost_2_shot.yaml) | $32*224^2$ | 2     | UCF-101                | 90.3       | [Link](https://huggingface.co/spaces/tomchen-ctj/OST/blob/main/checkpoints/few_shot/ucf/k2_few_shot_ucf_OST_k400_ft.pth) |
| [OST](configs/few_shot/finetuning_few_shot/ucf101/16_32_ost_4_shot.yaml) | $32*224^2$ | 4     | UCF-101                | 92.6       | [Link](https://huggingface.co/spaces/tomchen-ctj/OST/blob/main/checkpoints/few_shot/ucf/k4_few_shot_ucf_OST_k400_ft.pth) |
| [OST](configs/few_shot/finetuning_few_shot/ucf101/16_32_ost_8_shot.yaml) | $32*224^2$ | 8     | UCF-101                | 94.4       | [Link](https://huggingface.co/spaces/tomchen-ctj/OST/blob/main/checkpoints/few_shot/ucf/k8_few_shot_ucf_OST_k400_ft.pth) |
| [OST](configs/few_shot/finetuning_few_shot/ucf101/16_32_ost_16_shot.yaml) | $32*224^2$ | 16    | UCF-101                | 96.2       | [Link](https://huggingface.co/spaces/tomchen-ctj/OST/blob/main/checkpoints/few_shot/ucf/k16_few_shot_ucf_OST_k400_ft.pth) |
| [OST](configs/few_shot/finetuning_few_shot/ssv2/16_32_ost_2_shot.yaml) | $32*224^2$ | 2     | Something-Something V2 | 8.0        | [Link](https://huggingface.co/spaces/tomchen-ctj/OST/blob/main/checkpoints/few_shot/ssv2/k2_few_shot_ssv2_OST_k400_ft.pth) |
| [OST](configs/few_shot/finetuning_few_shot/ssv2/16_32_ost_4_shot.yaml) | $32*224^2$ | 4     | Something-Something V2 | 8.9        | [Link](https://huggingface.co/spaces/tomchen-ctj/OST/blob/main/checkpoints/few_shot/ssv2/k4_few_shot_ssv2_OST_k400_ft.pth) |
| [OST](configs/few_shot/finetuning_few_shot/ssv2/16_32_ost_8_shot.yaml) | $32*224^2$ | 8     | Something-Something V2 | 10.5       | [Link](https://huggingface.co/spaces/tomchen-ctj/OST/blob/main/checkpoints/few_shot/ssv2/k8_few_shot_ssv2_OST_k400_ft.pth) |
| [OST](configs/few_shot/finetuning_few_shot/ssv2/16_32_ost_16_shot.yaml) | $32*224^2$ | 16    | Something-Something V2 | 12.6       | [Link](https://huggingface.co/spaces/tomchen-ctj/OST/blob/main/checkpoints/few_shot/ssv2/k16_few_shot_ssv2_OST_k400_ft.pth) |

## Citation

If you find this work useful, please consider citing our paper! ;-)

```
@article{
    chen2023ost,
    title={OST: Refining Text Knowledge with Optimal Spatio-Temporal Descriptor for General Video Recognition},
    author={Tongjia Chen, Hongshan Yu, Zhengeng Yang, Zechuan Li, Wei Sun, Chen Chen.},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2024}, 
 }
```

## Acknowledgment

The work was done while Tongjia was a research intern mentored by [Chen Chen](https://www.crcv.ucf.edu/chenchen/). We thank [Ming Li (UCF)](https://liming-ai.github.io/) and [Yong He (UWA)](https://scholar.google.com/citations?user=6mXfrTEAAAAJ&hl=en&oi=ao) for proof-reading and discussion.

This repository is built upon portions of [ViFi-CLIP](https://github.com/muzairkhattak/ViFi-CLIP), [MAXI](https://github.com/wlin-at/MAXI), and [Text4Vis](https://github.com/whwu95/Text4Vis/). We sincerely thank the authors for releasing their code.

