import synaption #import synaption module


net1 Net() #create new neural net
net1.add_layer(10,ReLU, hidden) #add 2 hidden layers to net1 with 10 nodes, and ReLU activation
net1.add_layer(10, ReLU, hidden)
net1.add_layer(2, Sigmoid, output) #add output later to node1, with 2 nodes and Sigmoid activation


tensorArray = [] #create a list of tensors for training
for i in range(20):
    Tensor addTensor #creates a new tensor
    addTensor.input(1,2,3,4,5,6,7,8,9,10) #creates the inputs for the tensor
    addTensor.output(1,2) #creates the labels for the tensor
    tensorArray.append(addTensor) #adds the tensor to the list


net1.train(tensorArray, 200, 0.1) #trains the net on the list of tensors, learning rate of 0.1
net1.save() #saves the net for later use
