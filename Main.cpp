#include "Node.hpp"
#include "Tensor.hpp"

int main() {
    using namespace nn;

    // Create two nodes: input and output
    Node input_node(
        2,                          // 2 inputs
        ActivationFunction::ReLU,  // Activation function
        "InputLayer",              // Layer name
        "InputNode1",              // Node name
        NodeType::Hidden           // Node type
    );

    Node output_node(
        1,                          // 1 input (output of input_node)
        ActivationFunction::Sigmoid, // Activation function
        "OutputLayer",              // Layer name
        "OutputNode1",              // Node name
        NodeType::Output            // Node type
    );

    // Connect the input node to the output node

    input_node.point_node(&output_node);

    // Feed dummy input data into input_node
    std::vector<double> dummy_input = { 0.5, -0.2 };




    // Set input_node inputs and activate
    input_node.inputs = dummy_input;

    input_node.print_parameters();
    output_node.print_parameters();

    std::cout << '\n';


    double input_output = input_node.activate();

    std::cout << "Output of InputNode: " << input_output << '\n';

    double input2 = output_node.inputs[0];

    std::cout << "Input of outputnode: " << input2 <<  '\n';

    double final_activate = output_node.activate();

    std::cout << "Node activation: " << final_activate << std::endl;


    output_node.backpropagate(1,0.5,10);
    input_node.backpropagate(.5, 10);


    input_node.backpropagate(.5, 10);

    std::cout << '\n';

    input_node.print_parameters();
    output_node.print_parameters();


    std::cout << "Press any key to continue\n";
    std::cin.get();
    return 0;
}
