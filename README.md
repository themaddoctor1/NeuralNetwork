# Neural Net

### Written by Christopher Hittner

This implementation of a Neural Network was created to play with the idea of a Neural Network.
It comes with Neuron and NeuralNet classes, which are used to construct Neural Networks.

# Installation

To use the code, simply download the code and begin using it. To use neuralnet.py in your code,
all you have to do is import it. Whether you import the file or the functions in the file, you can do it using
```python
import neuralnet
```
or
```python
from neuralnet import *
```

# Usage

All of the code thus far is contained within neuralnet.py. The file contains two classes that are used to drive
the functionality of the networks.

## Neuron

A Neuron is an object that is the basic unit of the NeuralNet. A neuron has a set of predecessors and weights,
which ensure that each input has a specific weight. They also have a firing threshold, which requires that the
input be at least that amount before it can fire.

The Neuron constructor is as follows:
```python
def __init__(self, t = 0):
    """Creates a Neuron with the following properties:
    - No predecessors and no weights. For every predecessor,
      there will be exactly one weight to correspond to it.
    - Will be currently in the unfired state. This means that
      the weight of the output will have to be calculated.
    - A firing threshold of t. This means that the output will
      be 0 unless the input is greater than the output.

    Parameters:
        t - The minimum value required to fire the Neuron.
    """
```

A Neuron can be given predecessors by running ```addPredecessor(p, w)```, which requires p to be a Neuron
and w to be the numerical weight of the synapse between the two Neurons.

A Neuron can be fired by running ```fire()```. If the Neuron has fired without being reset, the Neuron
will return the last value calculated. In NeuralNet, this is done to avoid repeating calculations and
reduce runtime.

A Neuron can be reset using either ```reset()``` or ```resetAll()```. ```reset()`` will simply revert
the Neuron to its unfired state, while ```resetAll()``` will reset all of its children using ```resetAll()```
as long as the Neuron has at least one predecessor and the Neuron has fired. These are checked because input
and bias Neurons should not reset, and in NeuralNet, if a Neuron has not been fired, it can be presumed that
none of the Neuron's children have fired.

## NeuralNet
A NeuralNet represents a layered network of Neurons. These layers include an input and output layer, as well
as a series out hidden layers.

The NeuralNet constructor is as follows:
```python
def __init__(self, inSize, layers, outSize):
    """Creates a Neural Net with the following properties:
    - There are inSize input Neurons, plus one bias Neuron in the input layer.
    - There are layers hidden layers in the Neural Net, each of which will have
      exactly one bias Neuron preinstalled in the layer.
    - There are outsize Neurons in the output layer.

    Parameters:
	inSize  - The number of user controlled input Neurons.
	layers  - The number of hidden layers in the network.
 	outSize - The number of output Neurons.
```

A new layer can be added by calling ```addNewLayer()```. This will instantiate a layer adjacent to the
output layer, breaking any connection to the layer that was previously attached to the output. The function
will also generate a single bias Neuron on the layer, which will connect directly to the output layer.

A new Neuron can be added by calling ```addNewNeuron(layer)```. This requires the user to provide the layer
on which the Neuron is to be placed, starting at layer 0, which is adjacent to the input, and ending at 
```numHiddenLayers() - 1```, which is closest to the output layer. The Neuron will be connected to the 
layer above, and will be given predecessors from the layer below. If adjacent to the input or output, the
function will properly handle it.

A NeuralNet can be given an input by running ```setInput(idx, val)```, where ```idx``` is the input index between
0 and ```numInputs() - 1```, and ```val``` is the numerical value of the provided input.

A NeuralNet can be fired by running ```fire()```. This will run ```fire()``` to be run on all of the output Neurons.
If the output is required, the ``genOut()``` function can be used to fire the network, reset it, and return the
output values. If the user needs to manually reset the firing state, the ```reset()``` method can be used.

## Training

Although it is not perfect, the ```train()``` function can be used to train Neural Networks. The function
operates by iterating over every Neuron, and tweaking every weight, as well as the overall threshold in
order to obtain a better result. This is done by providing a set of training data, as well as an integer
number of iterations that the function should run through before finishing the training process.

The function head and documentation is as follows:
```python
def train(net, traindata, iterations):
    """Trains a neural network network of Neurons to operate correctly on a set of data.

    Arguments:
    net        - The Network to train.
                 NOTE: net will be modified in order to complete training.
    traindata  - The training data. Should be a list of pairs of tuples.
                 The tuples pairs consist of a set of inputs followed
                 by the correct outputs.
    iterations - The number of training rounds to perform.
    """
```

For testing purposes, several small data sets have been provided:
```python
ANDtree = [
    ([0,0],[0.0]),
    ([1,0],[0.0]),
    ([0,1],[0.0]),
    ([1,1],[1.0])
]

ORtree = [
    ([0,0],[0.0]),
    ([1,0],[1.0]),
    ([0,1],[1.0]),
    ([1,1],[1.0])
]

XORtree = [
    ([0,0],[0.0]),
    ([1,0],[1.0]),
    ([0,1],[1.0]),
    ([1,1],[0.0])
]

```

In order to create a training data set, one must follow the following rules:

- Each unit of training data must consist of a list of inputs and a list of outputs
- The number of inputs and outputs in each unit must be consistent throughout all of the data.

