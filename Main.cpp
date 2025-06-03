#include "Node.hpp"
#include <iostream>
#include <vector>

int main() {
    using namespace nn;

    // Create a node with 3 inputs, using Sigmoid activation, in "HiddenLayer1", named "Node1", and of type Hidden
    Node node(3, ActivationFunction::Sigmoid, "HiddenLayer1", "Node1", NodeType::Hidden);

    // Sample input vector
    std::vector<double> input = { 0.5, -0.3, 0.8 };

    // Forward activation
    double output = node.activate(input);
    std::cout << "Activation output: " << output << "\n";

    // Simulated error signal
    double error = 0.2;
    double learning_rate = 0.05;

    // Backpropagation step
    node.backpropagate(error, learning_rate, input);

    // Show parameters after one backpropagation step
    node.print_parameters();

    // Demonstrate safe backpropagation
    for (int i = 0; i < 15; ++i) {
        node.safe_backpropagate(error, learning_rate, input);
    }


    node.print_parameters();







    std::cout << "Press any key to continue\n";
    std::cin.get();
    return 0;
}
