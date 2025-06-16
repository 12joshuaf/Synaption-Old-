#include <iostream>
#include <vector>
#include "Layer.hpp"

using namespace nn;

int main() {
    std::cout << "=== Layer Test with Dummy Data ===\n";



    Layer hidden_layer(2, 3, ActivationFunction::ReLU, "Hidden", NodeType::Hidden);

    // Input vector (3 inputs)
    std::vector<double> input = { 1.0, 0.5, -0.5 };

    // Activate the hidden layer
    hidden_layer.activate(input);

    // Print hidden layer outputs
    std::cout << "Hidden layer outputs:\n";
    for (const auto& out : hidden_layer.get_outputs()) {
        std::cout << out << " ";
    }
    std::cout << "\n";

    // Create an output layer with 2 nodes (to match hidden layer output)
    Layer output_layer(2, 2, ActivationFunction::Sigmoid, "Output", NodeType::Output);

    // Activate output layer using output from hidden layer
    output_layer.activate(hidden_layer.get_outputs());

    std::cout << "Output layer predictions:\n";
    for (const auto& out : output_layer.get_outputs()) {
        std::cout << out << " ";
    }
    std::cout << "\n";

    // Dummy targets to test backprop
    std::vector<double> targets = { 0.0, 1.0 };

    // Backpropagate through output layer
    double learning_rate = 0.1;
    int saturation_threshold = 1000;

    std::cout << "Backpropagating output layer...\n";
    output_layer.backpropagate(targets, learning_rate, saturation_threshold);

    // Backpropagate through hidden layer
    std::cout << "Backpropagating hidden layer...\n";
    hidden_layer.backpropagate(learning_rate, saturation_threshold);

    std::cout << "Test complete.\n";

    std::cin.get();

    return 0;
}
