# Text to Image using DCGAN

This implementation is a PyTorch-based version of [Generative Adversarial Text-to-Image Synthesis paper](https://arxiv.org/abs/1605.05396). In this project, a Conditional Generative Adversarial Network (CGAN) is trained, leveraging text descriptions as conditioning inputs to generate corresponding images. The architecture of this model draws inspiration from DCGAN (Deep Convolutional Generative Adversarial Network).

## Requirements

- h5py==3.6.0
- numpy==1.21.5
- Pillow==10.0.0
- torch==2.0.0

## Dataset

We used [Caltech-UCSD Birds 200](http://www.vision.caltech.edu/visipedia/CUB-200.html) and text embeddings provided by [Reed Scott et al.](https://github.com/reedscot/icml2016)

## Repository

```
├── models
├     └──  dcgan_model.py
├── utils.py
├── data_util.py
├── requirements.txt
└──  DCGAN_Text2Image.ipynb
```

## Results

![](result/output_gif_20230719.gif)

## References

[1]  Generative Adversarial Text-to-Image Synthesis https://arxiv.org/abs/1605.05396

