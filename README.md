# PTQ4DM: Post-training Quantization on Diffusion Models    
Yuzhang Shang*, Zhihang Yuan*, Bin Xie, Bingzhe Wu, and Yan Yan    

The code for the Post-training Quantization on Diffusion Models, which has been accepted to CVPR 2023. [paper](https://arxiv.org/abs/2211.15736)

<img src="activation_hist.png" width="700">    
Key Obersevation: 

## Quick Start
First, download our repo:
```bash
git clone https://github.com/42Shawn/PTQ4DM.git
cd PTQ4DM
```
Then, run the DNTC script:
```bash
bash quant_sample_ddim_in_backword_DNTC.sh
```

**Demo Result**   
baseline (full-precision IDDPM) => 8-bit PTQ4DM    
           FID 21.7 => 24.3

# Reference
If you find our code useful for your research, please cite our paper.
```
@inproceedings{
shang2023ptqdm,
title={Post-training Quantization on Diffusion Models},
author={Yuzhang Shang and Zhihang Yuan and Bin Xie and Bingzhe Wu and Yan Yan},
booktitle={CVPR},
year={2023}
}
```

**Related Work**    
Our repo is modified based on the Pytorch implementations of Improved Diffusion ([IDDPM](https://github.com/openai/improved-diffusion), ICML 2021) and QDrop ([QDrop](https://github.com/wimh966/QDrop), ICLR 2022). Thanks to the authors for releasing their codebases!
