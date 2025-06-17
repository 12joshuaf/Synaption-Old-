import synaption #import synaption module


net1 = Net() #create new neural net
net1.add_layer(3,ReLU, hidden) #add 2 hidden layers to net1 with 10 nodes, and ReLU activation
net1.add_layer(3, ReLU, hidden)
net1.add_layer(2, Sigmoid, output) #add output later to node1, with 2 nodes and Sigmoid activation


tensorArray = [] #create a list of tensors for training
for i in range(20):
    addTensor = Tensor([1,2,3] , [4,5]) #create tensor that has inputs of 1,2,3 and labels of 4 and 5
    tensorArray.append(addTensor) #adds the tensor to the list


net1.activate(tensorArray[1]) #forward pass through net on first tensor
net1.backpropogate(tensorArray[1], 0.1) #backpropogation on net1 with learning rate of 0.1 on first tensor in array
net1.save() #saves the net for later use
