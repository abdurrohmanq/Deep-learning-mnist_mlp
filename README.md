[README.md](https://github.com/user-attachments/files/32416701/README.md)
# MNIST Handwritten Digit Classification

A feedforward neural network built with PyTorch that classifies handwritten digits from the MNIST dataset. This project covers the complete pipeline — data loading and normalization, model definition, training loop, and error analysis.

**Test accuracy: 94.68%** (5 epochs, plain SGD)

---

## Overview

The goal was to build a digit classifier from scratch and, just as importantly, to understand what every line of the training loop actually does. The model is a simple multilayer perceptron (MLP): no convolutions, no pretrained weights, no augmentation.

The result is deliberately a baseline. A CNN reaches ~99% on the same data — the gap between these two numbers is the point of the error analysis below.

## Dataset

[MNIST](http://yann.lecun.com/exdb/mnist/) — 70,000 grayscale images of handwritten digits, 28×28 pixels.

| Split | Images | Source |
|---|---|---|
| Train | 60,000 | US Census Bureau employees |
| Test | 10,000 | High school students |

The train and test sets were written by different groups of people, which makes the test set a genuinely independent evaluation rather than a random split of the same population.

Class distribution is nearly balanced (5,421–6,742 samples per digit), so plain accuracy is a meaningful metric here.

## Preprocessing

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

`ToTensor()` does three things: converts PIL images to tensors, scales pixels from `[0, 255]` to `[0.0, 1.0]`, and reorders dimensions from `(H, W, C)` to `(C, H, W)`.

`Normalize()` centers pixels using MNIST's precomputed mean and standard deviation. Without it the same model converges noticeably slower — roughly 90% instead of 94.7% after 5 epochs.

## Architecture

```python
model = nn.Sequential(
    nn.Flatten(),           # [64, 1, 28, 28] -> [64, 784]
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
```

| Layer | Output shape | Parameters |
|---|---|---|
| `Flatten` | `[64, 784]` | 0 |
| `Linear(784, 128)` | `[64, 128]` | 100,480 |
| `ReLU` | `[64, 128]` | 0 |
| `Linear(128, 10)` | `[64, 10]` | 1,290 |
| **Total** | | **101,770** |

Two design notes:

- **No `Softmax` at the output.** `nn.CrossEntropyLoss` applies log-softmax internally. Adding an explicit softmax layer applies it twice, which weakens gradients and slows convergence.
- **`ReLU` is what makes depth meaningful.** Two stacked `Linear` layers without a nonlinearity collapse algebraically into a single linear layer — the model would be no more expressive than logistic regression.

## Training setup

| Setting | Value |
|---|---|
| Loss | `nn.CrossEntropyLoss` |
| Optimizer | `optim.SGD`, `lr=0.01`, no momentum |
| Batch size | 64 |
| Epochs | 5 |
| Weight updates | 938 batches × 5 epochs = 4,690 |

## Results

### Training loss

| Epoch | Average loss |
|---|---|
| 1 | 0.5913 |
| 2 | 0.2988 |
| 3 | 0.2514 |
| 4 | 0.2195 |
| 5 | 0.1953 |

The curve is still descending at epoch 5 — the model has not converged. This is expected with plain SGD at `lr=0.01`; more epochs or added momentum both close the gap.

### Test performance

**Accuracy: 94.68%** — 532 of 10,000 images misclassified.

| Digit | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 0 | 0.962 | 0.985 | 0.973 | 980 |
| 1 | 0.970 | 0.984 | 0.977 | 1135 |
| 2 | 0.947 | 0.941 | 0.944 | 1032 |
| 3 | 0.950 | 0.926 | 0.938 | 1010 |
| 4 | 0.933 | 0.957 | 0.945 | 982 |
| 5 | 0.924 | 0.939 | 0.932 | 892 |
| 6 | 0.939 | 0.956 | 0.948 | 958 |
| 7 | 0.948 | 0.935 | 0.941 | 1028 |
| 8 | 0.941 | 0.922 | 0.932 | 974 |
| 9 | 0.947 | 0.919 | 0.933 | 1009 |

### For comparison

| Model | Test accuracy |
|---|---|
| Random guess | 10% |
| Logistic regression | ~92% |
| **This MLP (5 epochs, SGD)** | **94.68%** |
| Same MLP, 20 epochs + momentum | ~97.5% |
| CNN | ~99.2% |
| Human | ~99.8% |

## Error analysis

Errors are not spread evenly across classes — they cluster into specific confusable pairs.

| Pair | Errors (both directions) | Why |
|---|---|---|
| 9 ↔ 4 | 45 | An open-topped 9 is geometrically a 4 |
| 3 ↔ 5 | 34 | Same skeleton: top horizontal stroke plus lower curve |
| 7 ↔ 2 | 29 | A looped 7 and a flat-based 2 overlap |
| 8 ↔ 3 | 29 | An 8 with an open left half reads as a 3 |

Digits 0 and 1 have the highest recall (0.985 and 0.984) — their shapes resemble nothing else in the set. Digit 9 has the lowest recall (0.919), and digit 4 correspondingly has low precision (0.933): the misread 9s land in the 4 column.

### The structural limitation

`nn.Flatten()` destroys spatial information. After flattening, the model sees 784 independent numbers with no knowledge that pixel 47 sits next to pixel 48. As a consequence:

- A digit shifted a few pixels to the right becomes a completely different input vector
- The model has no concept of "loop" or "vertical stroke" — only "how much light falls at position *n*"
- What it learns is each digit's *average pixel layout*, not its *shape*

Visualizing the effective weight template per class makes this concrete: the template for `1` is a fixed bright vertical band in the center columns with negative weights on either side. Any 1 written slightly off-center misses that band.

This is exactly what convolutional layers fix — `nn.Conv2d` slides a small filter across the whole image, so a feature is detected regardless of where it appears.

## Project structure

```
.
├── mnist_mlp.ipynb        # Full pipeline: data, model, training, evaluation
├── requirements.txt
└── README.md
```

## Requirements

```
torch
torchvision
matplotlib
scikit-learn
seaborn
numpy
```

## Usage

```bash
git clone https://github.com/abdurrohmanq/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
jupyter notebook mnist_mlp.ipynb
```

The dataset downloads automatically on first run (`download=True`) into `./data`. The notebook also runs unmodified in Google Colab — enable a GPU runtime for faster training, though this model trains in a couple of minutes on CPU.

## Key takeaways

- A single neuron with a sigmoid activation is mathematically identical to logistic regression. Deep learning extends this idea rather than replacing it.
- Class labels carry no inherent meaning to the model. `labels = 1` means "index 1 is correct", not "the digit one". Shuffling the label mapping produces an equally accurate model that simply reports predictions at different indices.
- Normalization is not cosmetic — it accounted for roughly 4 percentage points of accuracy here.
- `optimizer.zero_grad()` is mandatory: PyTorch accumulates gradients rather than overwriting them.
- Deep learning is not a universally stronger alternative to classical ML. On tabular data, gradient boosting typically outperforms a neural network. Images are where the architecture earns its keep — and even then, only when the architecture matches the data.

## Next steps

- [ ] Add a validation split so hyperparameters are tuned without touching the test set
- [ ] Compare optimizers: SGD vs SGD + momentum vs Adam
- [ ] Implement a CNN and compare the confusion matrices pair by pair
- [ ] Apply `RandomRotation` augmentation to reduce the 9 ↔ 4 confusion

## Author

**Abdurahmon Qodirov**

- GitHub: [@abdurrohmanq](https://github.com/abdurrohmanq)
- LinkedIn: [abdurrohman-qodirov](https://linkedin.com/in/abdurrohman-qodirov)

## License

MIT
