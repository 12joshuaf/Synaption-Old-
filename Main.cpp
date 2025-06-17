#include "Net.hpp"



int main() {
    nn::Net net;

    net.add_layer(4, 3, nn::ActivationFunction::ReLU, nn::NodeType::Hidden);
    net.add_layer(2, 4, nn::ActivationFunction::Sigmoid, nn::NodeType::Output);

    std::vector<nn::Layer*> layers = {
        net.get_layer(0),
        net.get_layer(1)
    };

    std::cout << "Before training:\n";
    net.print_parameters();

    std::vector<double> input = { 0.5, 0.1, -0.4 };

    layers[0]->activate(input);
    std::vector<double> hidden_output = layers[0]->get_outputs();

    layers[1]->activate(hidden_output);
    std::vector<double> output = layers[1]->get_outputs();

    std::vector<double> target = { 1.0, 0.0 };

    double learning_rate = 0.1;
    int saturation_threshold = 5;

    layers[1]->backpropagate(target, learning_rate, saturation_threshold);
    layers[0]->backpropagate(learning_rate, saturation_threshold);

    std::cout << "\nAfter training:\n";
    net.print_parameters();

    std::cout << "Backpropagation completed.\n";

    std::cout << "Press any key to continue...";
    std::cin.get();

    return 0;
}

