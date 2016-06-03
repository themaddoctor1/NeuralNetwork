import random, math

"""A Neuron is a type that has a set of output weights, which
are used to calculate a net output depending on the output
of the Neuron's predecessors.
"""
class Neuron:
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
        # The firing weight of the Neuron.
        self.weights = []   # The weights of the connections between the Neurons.
        self.lastOut = 0    # The value outputed. This is only valid if tick is True.
        self.preds = []     # The Neuron's predecessors.
        self.tick = False   # Whether or not the Neuron has computed a firing weight.
        self.threshold = 0  # The activation value required to fire.

    def __str__(self):
        res = "--" + str(self.weights) + "--[(O_O)]--{" + str(len(self.preds)) + "}-->"
        if(self.tick):
            res += str(self.lastOut)
        return res

    def addPredecessor(self, p, w):
        """Adds a predecessor p with edge weight w."""
        self.preds.append(p)
        self.weights.append(w)
    
    def fire(self):
        """Fires the Neuron, and returns the output weight.
        Will modify value of stored last value returned.
        Returns the output value.
        """
        # If ticked, return the last output.
        if self.tick:
            return self.lastOut
        
        # The sum of all of the input weights.
        summ = 0

        # For each predecessor, add the output to
        # the sum of the input weights.
        for i in range(len(self.preds)):
            summ += self.preds[i].fire() * self.weights[i] # Expected to be O(|N|)

        # Activation function goes here. It should be a function of the input (stored in summ).
        out = summ # 1 / (1 + math.exp(-summ))
        if out < self.threshold:
            out = 0
        
        
        # Trigger the Neuron
        self.trigger(out)
        
        # The result is the value of the sigmoid function sig(x_i * w_i)
        return out

    def trigger(self, mag):
        """Changes the lastOut value to a particular value, and marks the Neuron as fired."""
        self.tick = True
        self.lastOut = mag
    
    def reset(self):
        """Resets the Neuron to the unfired state."""
        self.tick = False
        self.lastOut = 0

    def resetAll(self):
        """Resets the Neuron and all of its children, if needed.
        If this Neuron is at the unfired state, it is assumed that
        all of its children meet the same condition.
        
        Reset should occur if
          (1) It has been activated (therefore should be turned off)
          (2) It has predecessors (implies it is neither an input nor bias)
        """
        
        if self.tick and len(self.preds) > 0:
            self.reset()
            for n in self.preds:
                n.resetAll()




class NeuralNet:
    """A NeuralNet is a collection of Neurons. The Neurons are organized into
    layers, which are interconnected by synapses with initially random weights.
    """
    
    def __init__(self, inSize, layers, outSize):
        
        # Sets up the input layer
        self.inLayer = []
        for i in range(inSize + 1):
            n = Neuron(random.uniform(0, 1))
            self.inLayer.append(n)
        # Index 0 of the inputs must be a bias Neuron
        self.inLayer[0].trigger(1)

        # Creates a set of hidden layers. At the start, there are no layers
        self.hiddenLayers = []

        # Creates all of the requested hidden layers
        for i in range(0, layers):
            #Creates an empty list and adds it to the layer list.
            self.hiddenLayers.append([])
            
            # Generates a new Neuron and then adds it
            n = Neuron(random.uniform(0, 1))
            self.hiddenLayers[i].append(n)
            
            # Given that n is at index 0, it must be treated as a bias Neuron.
            n.trigger(1)
                    
        # Sets up the output layer
        self.outLayer = []
        for i in range(outSize):
            n = Neuron(random.uniform(0, 1))
            self.outLayer.append(n)
            n.addPredecessor(self.hiddenLayers[layers-1][0], random.uniform(-1, 1))
            
            
    def __str__(self):
        res = "Net size: " + str(self.size()) + "\n"

        res += "IN\n"
        for i in range(len(self.inLayer)):
            res += " [" + str(i) + "] " + str(self.inLayer[i]) + "\n"
            
        for i in range(len(self.hiddenLayers)):
            res += "HL_" + str(i) + "\n"
            for j in range(len(self.hiddenLayers[i])):
                res += " [" + str(j) + "] " + str(self.hiddenLayers[i][j]) + "\n"
        
        res += "OUT\n"
        for i in range(len(self.outLayer)):
            res += " [" + str(i) + "] " + str(self.outLayer[i]) + "\n"
        return res
    
    def addNewLayer(self):    
        """Adds a new layer to the top of the hidden layer array. This method is
        used specifically by the constructor to perform operations.
        """
        
        #Creates an empty list and adds it to the layer list.
        self.hiddenLayers.append([])
        
        # Adds a Neuron to the newly added layer.
        self.addNewNeuron(self.numHiddenLayers()-1)

        # Grabs the new layer.
        new = hiddenLayers[hLayers-1];

        # The sole Neuron should have no weights for now.
        new[0].weights = []
        # Furthermore, the first Neuron should always be a bias node.
        new[0].trigger(1)

        # Index 0 of any hidden layer is considered to be a bias node.
        
        # All of the Neurons on the previous layers should be predecessors.
        # hLayers = self.numHiddenLayers()
        # for n in self.hiddenLayers[hLayers-2]:
        #     new[0].addPredecessor(n, random.uniform(-1, 1))

        # Resets the inputs for the output layer.
        self.outLayer = []
        for n in self.outLayer:
            n.addPredecessor(new[0], random.uniform(-1, 1))
    
    def addNewNeuron(self, layer):
        """Adds a Neuron to a particular layer. This method is used
        specifically by the constructor to perform operations.
        """

        # Generates a new Neuron and then adds it
        n = Neuron(random.uniform(0, 1))
        self.hiddenLayers[layer].append(n)
        
        # Then, depending on what layer the Neuron is on,
        # set up the proper predecessors
        if layer > 0: # The predecessors are on the input layer.
            for m in self.hiddenLayers[layer-1]:
                n.addPredecessor(m, random.uniform(-1, 1))
        else: # The predecessors are in the hidden layers.
            for p in self.inLayer:
                n.addPredecessor(p, random.uniform(-1, 1))

        # Finally, have all of the Neurons in the layer above point to this Neuron.
        if layer < self.numHiddenLayers() - 1: # If the layer is not adjacent to the output layer
            for m in range(1, len(self.hiddenLayers[layer+1])):
                self.hiddenLayers[layer+1][m].addPredecessor(n, random.uniform(-1, 1)) # Add the new Neuron as pred to the above.
        else: # Otherwise, the new Neuron must be a predecessor of the Neuron.
            for m in self.outLayer:
                m.addPredecessor(n, random.uniform(-1, 1)) # Add the Neuron as pred to outputs

    def setInput(self, idx, val):
        """Sets the given input node to a value."""
        self.inLayer[idx+1].trigger(val)
                
    def fire(self):
        """Fires the Neural Network."""
        for n in self.outLayer:
            n.fire()

    def genOut(self):
        """Generates the output results of the network,
        and then resets the network.
        
        Returns: a float array of outputs.
        """
        
        self.fire()
        res = []
        for n in self.outLayer:
            res.append(n.fire())
        self.reset()
        return res

    def reset(self):
        """Resets the entire NeuralNet to the unfired state."""
        for n in self.outLayer:
            n.resetAll()

    def numInputs(self):
        """Computes the number of inputs.
        This does not include the bias node.
        """
        return len(self.inLayer) - 1

    def numOutputs(self):
        return len(self.outLayer)
    
    def numHiddenLayers(self):
        """Determines and returns the number of hidden layers."""
        return len(self.hiddenLayers)
    
    def size(self):
        """Determines and returns the number of Neurons in the network."""
        res = len(self.inLayer) + len(self.outLayer)
        for l in self.hiddenLayers:
            res += len(l)
        return res;

    def hiddenNetSize(self):
        res = 0;
        for i in range(self.numHiddenLayers()):
            res += self.hiddenLayerSize(i)
        return res;
    
    def hiddenLayerSize(self, layer):
        return len(self.hiddenLayers[layer])




def __genError(net, traindata):
    err = 0
    for pair in traindata:
        # Calculate the output for this efficiency
        for inIdx in range(len(pair[0])):
            net.setInput(inIdx, pair[0][inIdx])
                            
        # Grabs the result of the operation, and computes the error.
        runRes = net.genOut()
    
        for outIdx in range(len(pair[1])):
            err += math.pow(pair[1][outIdx] - runRes[outIdx], 2)

    # Finalize the error calculation
    err /= (net.numInputs() * len(traindata))
    return err
    
    
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
    
    # Computes a certain number of training rounds.
    for it in range(iterations):
        # In each iteration, we will improve every Neuron in every layer (aside from input)
        for n in ([n for layer in net.hiddenLayers for n in layer]):
            # For the Neuron, we want to run the inputs on the network a bunch of
            # times in order to find the best possible weights for the Neuron.

            # Improves the weight
            for edgeIdx in range(len(n.weights)):
                
                bestErr = float("inf")
                bestWeight = 0
                
                # Iterates until a sufficient amount of data has been collected.
                for i in range(64): # For now, I have elected to test 64 points.
                    
                    err = __genError(net, traindata)
                        
                    # Inserts a coordinate into the graph.
                    if bestErr >= err:
                        bestErr = err
                        bestWeight = n.weights[edgeIdx]
                    
                    # Calculate a new weight to test
                    
                    # This method would be ideal, but it causes values to skyrocket and become unusable.
                    newVal = random.uniform(-1, 1) # Weights should be between -1 and 1.
                    
                    n.weights[edgeIdx] = newVal
                
                n.weights[edgeIdx] = bestWeight

            # Computes a better threshold
            bestErr = float("inf")
            bestThresh = 0
            
            # Iterates until a sufficient amount of data has been collected.
            for i in range(64): # For now, I have elected to test 64 points.
                
                err = __genError(net, traindata)
                        
                # Inserts a coordinate into the graph.
                if bestErr > err:
                    bestErr = err
                    bestThresh = n.threshold
                                
                # This method would be ideal, but it causes values to skyrocket and become unusable.
                newVal = random.uniform(-net.hiddenNetSize(), net.hiddenNetSize()) # Thresholds should be based on the hidden net
                    
                n.threshold = newVal
                
            n.threshold = bestThresh

            
    return net

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
