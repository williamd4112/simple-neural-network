# Introduction
This project is a python implementation of neural network a.k.a. multi-layer perceptron.  In this project, I experimented two architecture of neural networks (2 hidden layer and 1 hidden layer) and all input data are pre-processed with PCA method.  Both feed forward and back propgation are included in the implementation.  In backpropgation, I apply gradient in mini-batch gradient descent method.

# Dependencies
- numpy v1.12
- tqdm
- OpenCV

# Dataset
Database of Faces ( AT&T Laboratories Cambridge)    
Reference : http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html

# Results
Best error rate of each model

|1-hidden layer   |  2-hidden layer|
|:-:|:-:|
|0.018333   |  0.01 |

# Visualization (Decision boundary)
|1-hidden layer   |  2-hidden layer |
| ------------- |:------------:|
|![h1](/doc/nn_sigmoid.png)|![h2](/doc/nn_relu.png)|

# To train the model (examples)
Training scripts use default training data in data/class*.npy and default training hyperparameters. If you want to use your own data, please see the manual of main.py
```
./train_nn.sh "{fraction of class 1},{fraction of class 2},{fraction of class 3}" {epoch} {batch_size} {learning rate}(2-hidden layer)
./train_nn_2.sh "{fraction of class 1},{fraction of class 2},{fraction of class 3}" {epoch} {batch_size} {learning rate}(1-hidden layer)
```

# To validate the model (examples)
```
./validate_nn.sh "{fraction of class 1},{fraction of class 2},{fraction of class 3}" {epoch} {batch_size} {learning rate} (2-hidden layer)
./validate_nn_2.sh  "{fraction of class 1},{fraction of class 2},{fraction of class 3}" {epoch} {batch_size} {learning rate} (1-hidden layer)
```

# To test the model (examples)
```
./test_nn.sh {result output}
```

# To run demo
This script pre-processes the demo images in Demo directory and run test script, save result to result directory.
```
./demo_nn.sh (2-hidden layer)
./demo_nn_2.sh (1-hidden layer)
```
