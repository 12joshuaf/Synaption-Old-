#include "Net.hpp"
#include <iostream>
#include <vector>

int main() {
    using namespace nn;

    // Build the network
    Net net;

    // Input layer (manually using Hidden type since Input may not do much yet)
    net.add_layer(2, 2, ActivationFunction::Sigmoid, NodeType::Hidden);

    // Hidden layer
    net.add_layer(3, 2, ActivationFunction::Sigmoid, NodeType::Hidden);

    // Output layer
    net.add_layer(1, 3, ActivationFunction::Sigmoid, NodeType::Output);

    // Training data (XOR-like)
    std::vector<std::vector<double>> inputs = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };

    std::vector<std::vector<double>> targets = {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };

    const double learning_rate = 0.1;
    const int saturation_threshold = 10;

    // Train the network
    std::cout << "Training network...\n";
    for (int epoch = 0; epoch < 1000; ++epoch) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            net.activate(inputs[i]);
            net.backpropagate(targets[i], learning_rate, saturation_threshold);
        }
    }

    std::cout << "Training complete.\n\n";

    // Test the network
    std::cout << "Testing trained network:\n";
    for (size_t i = 0; i < inputs.size(); ++i) {
        net.activate(inputs[i]);
        std::vector<double> output = net.layers.back()->get_outputs();
        std::cout << "Input: (" << inputs[i][0] << ", " << inputs[i][1] << ") -> Output: " << output[0] << "\n";
    }

    // Print network parameters (verbose off)
    std::cout << "\nNetwork parameters:\n";
    net.print_parameters(false);

    // Save network
    net.save_net("my_trained_net");


    std::cout << "Press any key to continue\n";
    std::cin.get();

    return 0;
}
