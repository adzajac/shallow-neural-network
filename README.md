# shallow neural network
Shallow neural network in Python3 and NumPy built from scratch


# Example of use

Create a NeuralNet class instance:
```python
net = NeuralNet(input_size=INPUT_SIZE, neurons_num=NUMB_OF_NEURONS)
```

Prepare X_train and Y_train with NumPy.

Train neural network:
```python
net.train(X_train, Y_train, epoch_num=100, batch_size=10, cost_fun="cross_entropy", lambd=0.01)
```
