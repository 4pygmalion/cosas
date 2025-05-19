# COSAS – Cross-Scanner Adenocarcinoma Segmentation  
*A top-10 solution in the MICCAI 2024 COSAS challenge, now published as a full paper at **MIE 2025***  

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---

## Table of contents
1. [About the project](#about-the-project)  
2. [Method overview](#method-overview)  
3. [Quick start](#quick-start)  
4. [Results](#results)
5. [Citations](#cite)

---

## About the project
COSAS explores **domain-shift–robust histopathology** by jointly learning **tumour segmentation** and **stain separation**.  
Our joint multi-task U-Net:  

* isolates stain *matrix* & *density* to handle colour variation,  
* boosts mean Dice to **0.898** and IoU to **0.816** on internal validation,  
* generalises to six unseen scanners with mean Dice/IoU **0.792**. :contentReference[oaicite:1]{index=1}  

For challenge details see the official [COSAS grand-challenge page](https://cosas.grand-challenge.org/). :contentReference[oaicite:2]{index=2}  

---

## Method overview
<img src="docs/architecture.png" alt="Multi-decoder U-Net" width="600"> ``` Our architecture couples a pretrained **EfficientNet-B7 encoder** with **two decoders**: * **Stain-matrix decoder** – learns stain colour bases. * **Stain-density decoder** – captures tissue structure. Features are fused for segmentation; training minimises a weighted sum of reconstruction and segmentation losses (α ≈ 0.3). See the full paper for details. :contentReference[oaicite:4]{index=4}


## Quick start
Requires Docker ≥ 24 & make.
# 1. Build an inference image around your trained checkpoint
```bash
make build MODEL_PATH=/path/to/model.pth
```

# 2. Run a quick smoke-test on sample images
```bash
make test_run
```

To export the container:
```bash
make save
```


## Results
Dataset / split	Dice	IoU
COSAS internal (3 scanners, 4-fold CV)	0.898	0.816
COSAS external (6 scanners)	0.792	0.792

The submitted model ranked top-10 on the COSAS final leaderboard (Task 2).

## Cite
@article{Kim2025COSAS,
  author    = {Ho Heon Kim and Won Chan Jeong and Youngjin Park and Young Sin Ko},
  title     = {Understanding Stain Separation Improves Cross-Scanner Adenocarcinoma Segmentation with Joint Multi-Task Learning},
  journal   = {Studies in Health Technology and Informatics},
  volume    = {327},
  pages     = {53--57},
  year      = {2025},
  doi       = {10.3233/SHTI250272},
  publisher = {IOS Press}
}



