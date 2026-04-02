# QCFS-based TPP

The code reference of the proposed TPP method.

The TPP design is in the `class IF` in `Models/layer.py`.

# How to run?

1. train the QCFS-based ReLU ANN models on CIFAR-100

```bash
bash cifar_train.sh
```

2. Conduct ANN-to-SNN conversion and evaluate the SNN models

```bash
bash cifar_test.sh
```
