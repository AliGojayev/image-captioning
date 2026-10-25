# Image Captioning

Image captioning in PyTorch using a pretrained VGG-16 encoder, spatial attention, and an LSTM decoder. Trained locally on an M4 Mac with 16 GB unified memory.

## Model

The frozen VGG-16 encoder extracts a 7 × 7 × 512 feature map. At each decoding step, additive attention computes a weighted context vector over the 49 spatial locations. The context is projected to 128 dimensions and concatenated with a 128-dimensional word embedding as input to an LSTM with 256 hidden units.

Training uses teacher forcing, cross-entropy loss with padding ignored, Adam, dropout of 0.3 on word embeddings and decoder outputs, and gradient clipping. A learning-rate scheduler reduces the rate when validation loss plateaus. Early stopping preserves the checkpoint with the lowest validation loss. Captions are generated using beam search with beam size 3 and length normalization of 0.6.

## Data and results

The project uses Flickr8k, with five reference captions per image. The experiments use 6,000 training images and a fixed subset of 500 validation images from the official development split. The vocabulary is built from training captions, retaining words appearing at least twice.

| Validation metric | Score |
|---|---:|
| BLEU-1 | 57.52 |
| BLEU-4 | 17.27 |

Scores use NLTK corpus BLEU, multiplied by 100, with all five references per image. The validation set was used during development; these are not held-out test results.

The model can generate plausible descriptions of common scenes, but sometimes misses details, predicts incorrect objects, attributes, and actions.

## Future work

The model can still be improved in many aspects.
Potential improvements include a stronger encoder such as ResNet-101, higher-resolution spatial features, and additional regularization to reduce overfitting.

## References

- [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044) — basis for the attention mechanism; this project uses a simplified decoder.
- [Flickr8k](https://hockenmaier.cs.illinois.edu/8k-pictures.html) — images and reference captions.
- VGG-16 from TorchVision; pretrained weights supplied through the KAIST Introduction to Deep Learning course.
