# CNeuralNet

CNeuralNet is a simple neural network library written in C. It allows you to easily create and train neural networks for various simple machine learning tasks.

## Features
* Supports fully connected (dense) neural networks with any number of hidden layers and neurons per layer
* Uses backpropagation and gradient descent for training
* Provides a simple API for creating and training neural networks
* Comes with a simple example program that demonstrates how to use the library

## Requirements
A C compiler (e.g. GCC, Clang)
CMake (for building the library and example program)

## Getting started
To use CNeuralNet in your own project, follow these steps:

1. Clone the repository and navigate to the project directory:
```
git clone https://github.com/tffdev/CNeuralNet.git
cd CNeuralNet
```
2. Build the library and example program using CMake:
```
cmake .
make
```
3. Use the library in your own C program by adding the NeuralNetwork.h and NeuralNetwork.cpp files to your own project.

## Documentation
See the NeuralNetwork.h header file for the API - it should be straight forward, otherwise see the example program.

## License
CNeuralNet is released under the MIT License



