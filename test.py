from single_neural_network import NeuralNetwork
from numpy import array

if __name__ == "__main__": 
    neural_network = NeuralNetwork() 
      
    print ('Random weights at the start of training') 
    print (neural_network.weight_matrix) 
  
    train_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]]) 
    train_outputs = array([[0, 1, 1, 0]]).T 
  
    neural_network.train(train_inputs, train_outputs, 10000) 
  
    print ('New weights after training') 
    print (neural_network.weight_matrix) 
  
    # Test the neural network with a new situation. 
    print ("Testing network on new examples ->") 
    print (neural_network.forward_propagation(array([1, 0, 0])))    