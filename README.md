# Feedforward Neural Network
This project implements a feedforward neural network in C to classify handwritten digits from the MNIST dataset. The network is trained on the MNIST training set and can achieve **98% accuracy** on the test set.

## Features
- Written in pure C, no external ML libraries used
- Supports arbitrary multi-layer model architectures
- Forward and backward propagation implemented from scratch
- Matrix and vector operations have been vectorized with AVX-512
- Uses ReLU activation for hidden layers and softmax activation for the output layer
- Uses a step decay learning rate scheduler
- Reads standard MNIST files directly

## Usage
1. Build the project
```
cmake -DCMAKE_BUILD_TYPE=Release -S . -B cmake-build-release
cmake --build cmake-build-release --config Release
```

2. Run the program
```
cd cmake-build-release
./NeuralNetwork
```
The program is set up to train the network and then evaluate it on 10k test cases that the network has not seen before. The training takes around 4-5 seconds.

## Accuracy
With the default settings (3 layers: 128, 64, 10 neurons) the network can achieve up to 98% accuracy on the MNIST test set after 10 epochs of training.

## Future Improvements
- Implement mini-batches
- Parallelization for training

## Requirements
- C Compiler
- CMake
- Standard C Library
