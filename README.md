
# p4m-gconv

Jupyter notebooks with implementation details of p4m-gconv. See the below githubs for more information:

Tests

- [X] **Both** CNN and G-Conv CNN train on rotated and mirrored MNIST images

- [X] **Both** CNN and G-Conv CNN train on only upright MNIST images

- [X] Train CNN with rotated images, G-Conv CNN with only upright

## Repository Links

[tscohen/GrouPy](https://github.com/tscohen/GrouPy)

[adambielski/GrouPy](https://github.com/adambielski/GrouPy) (Forked)

[COGMAR/RotEqNet](https://github.com/COGMAR/RotEqNet)

## Dependencies

[tscohen/GrouPy](https://github.com/tscohen/GrouPy)

## Results

To get these results, I trained two seperate models (with similar numbers of parameters) on the MNIST dataset with the following modifications.

### Network descriptions

* A normal CNN with two convolution layers, and two hidden fully connected layers.
* A group equivariant model with two convolution layers and two hidden fully connected layers.

### Better comparisons

50 epochs of training on similar size (parameter number) networks. Batch size of 64, shuffled.
Always test the neural network on rotated MNIST (This applies the 4 rotations and 2 mirrors possible as
described in [this paper](https://arxiv.org/pdf/1602.07576.pdf))

| Model | Upright MNIST Training | Rotated MNIST Training |
|-------|------------------------|------------------------|
| CNN   | 31.63%                 | 93.36%                 |
| P4M   | 97.24%                 |    (Is equivariant)    |
| P4    | 97.4%                  |    (Is equivariant)    |

Standard deviations have not been calculuated yet but I assume they are quite small.