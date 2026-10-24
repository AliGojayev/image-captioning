# Image Captioning

An educational image-captioning project in PyTorch, planned around a frozen VGG-16 image encoder and a trainable LSTM caption decoder.

## Current progress

- Loaded the original pretrained VGG-16 checkpoint provided for the KAIST Introduction to Deep Learning course.
- Kept and froze the convolutional feature extractor.
- Verified a forward pass on Apple Silicon using the PyTorch MPS backend.
- Confirmed that a 224 x 224 input produces a 512 x 7 x 7 feature map and a 512-dimensional vector after spatial averaging.

The caption dataset, decoder, training, and evaluation are not implemented yet. The notebook currently uses a dummy input to check tensor dimensions; no captioning results are claimed.

## Setup

Create and activate a Python virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Place the original pretrained `pretrained_vgg.pt` checkpoint in `checkpoints/`. The weights are not included in this repository. This project currently expects the course-provided VGG-16 state dictionary, rather than an FCN segmentation checkpoint.

Open `image_captioning.ipynb` in VS Code and select the `.venv` Python kernel. Run cells in order. The encoder uses MPS when available and otherwise uses CPU.

## Planned work

- Prepare image-caption pairs with separate image-level training, validation, and test splits.
- Cache frozen image features.
- Implement and train an LSTM decoder with teacher forcing.
- Evaluate generated captions and document successes, failures, and limitations.

## Acknowledgments

The encoder uses the VGG-16 architecture supplied by TorchVision and pretrained weights supplied through the KAIST Introduction to Deep Learning course. This repository contains a separate captioning project; course assignment files and pretrained weights are not included.
