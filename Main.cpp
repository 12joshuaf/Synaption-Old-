#include "Node.hpp"
//example code for building a two node algorithm

int main() {
    using namespace nn;

    //create two nodes
    Node input_node(2, ActivationFunction::ReLU, "InputLayer", "InputNode1", NodeType::Hidden);
    Node output_node(1, ActivationFunction::Sigmoid, "OutputLayer", "OutputNode1", NodeType::Output);

    //connect the two nodes together
    input_node.point_node(&output_node);

    //dummy data, add tensors later
    std::vector<double> dummy_input = { 0.5, -0.2 };
    double learning_rate = 0.5;
    int saturation_threshold = 10;

    input_node.inputs = dummy_input;
    input_node.print_parameters();
    output_node.print_parameters();

    std::cout << '\n';


    //training the two nodes
    double out1 = input_node.activate();
    std::cout << "Output of InputNode: " << out1 << '\n';

    double input_to_output = output_node.inputs[0];
    std::cout << "Input to OutputNode: " << input_to_output << '\n';

    double final_output = output_node.activate();
    std::cout << "Final output: " << final_output << "\n\n";

    output_node.backpropagate(1.0, learning_rate, saturation_threshold);
    input_node.backpropagate(learning_rate, saturation_threshold);

    std::cout << "\nAfter Backpropagation:\n";
    input_node.print_parameters();
    output_node.print_parameters();

    std::cout << "Press Enter to exit.\n";
    std::cin.get();
    return 0;
}
