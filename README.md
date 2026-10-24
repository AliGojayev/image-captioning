# Image Captioning

Image captioning with VGG-16 and an LSTM decoder in PyTorch. In progress.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place `pretrained_vgg.pt` in `checkpoints/`, then open `image_captioning.ipynb` using the `.venv` kernel.

## Status

Frozen VGG-16 encoder implemented. Caption data preparation and the LSTM decoder are next.

## Credits

VGG-16 from TorchVision; pretrained weights provided by the KAIST Introduction to Deep Learning course.
