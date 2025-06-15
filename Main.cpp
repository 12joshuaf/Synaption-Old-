#include <iostream>
#include "Layer.hpp"

using namespace nn;

int main() {
    // Network architecture
    const int input_size = 5;                // Example input size
    const int hidden_layer_size = 10;
    const int num_hidden_layers = 3;
    const int output_size = 1;

    // Example activation function (replace with your actual enum or function)
    ActivationFunction activation = ActivationFunction::Sigmoid;

    // Create layers
    std::vector<Layer> network;

    // First hidden layer (connected to input)
    network.emplace_back(hidden_layer_size, input_size, activation, "Hidden_0", NodeType::Hidden);

    // Additional hidden layers
    for (int i = 1; i < num_hidden_layers; ++i) {
        network.emplace_back(hidden_layer_size, hidden_layer_size, activation, "Hidden_" + std::to_string(i), NodeType::Hidden);
    }

    // Output layer (connected to last hidden layer)
    network.emplace_back(output_size, hidden_layer_size, activation, "Output", NodeType::Output);

    // Dummy input for testing
    std::vector<double> input(input_size, 1.0); // e.g., [1.0, 1.0, ..., 1.0]

    // Forward pass
    std::vector<double> current_input = input;
    for (auto& layer : network) {
        layer.activate(current_input);
        current_input.clear();
        for (const auto& node : layer.get_nodes()) {
            current_input.push_back(node.get_last_output());
        }
    }

    // Print final output
    std::cout << "Network output:\n";
    for (double out : current_input) {
        std::cout << out << "\n";
    }
    std::cout << "\n" << "press any key to continue";
    std::cin.get();

    return 0;
}
