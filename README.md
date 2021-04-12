# "_Target Speaker Text-to-Speech Synthesis from a Face Image Reference using Global Style Token Embedding_"
## Training, Inference and Data Preparation Code

This repository contains the Notebooks used for data preparation and visualization. It also contains the code for training FaceNet and finetuning on style embeddings.

To run this project a bit of steup is required:
1. Clone this repository
```bash
$ git clone git@github.com:bjoernpl/ThesisFiles.git
$ cd ThesisFiles
```
2. Clone ESPnet fork from [here](https://github.com/bjoernpl/espnet)
```bash
$ git clone git@github.com:bjoernpl/espnet.git
$ cd espnet
$ pip install -e .
$ cd ../src
```
2. Install [ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)
```bash
$ git clone https://github.com/kan-bayashi/ParallelWaveGAN.git
$ cd ParallelWaveGAN
$ pip install -e .
$ cd ..
```
3. Install FaceNet fork from [here](https://github.com/bjoernpl/facenet-pytorch)
```bash
$ git clone git@github.com:bjoernpl/facenet-pytorch.git facenet_pytorch
```
4. Download finetuned FaceNet weights
```bash
$ wget https://www.dropbox.com/s/iyokqy9jmcob7nm/face_style_weights.ckpt -O  ./models/weights/face_style_weights.ckpt
```
