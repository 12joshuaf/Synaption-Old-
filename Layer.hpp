#pragma once

#include "Node.hpp" // Include the Node class definition
#include <vector>
#include <string>

namespace nn {

    class Layer {
    private:
        std::vector<Node> nodes; // Vector of nodes in the layer
        std::string layer_name;  // Name of the layer

    public:
        // Constructor to create a layer with a specified number of nodes
        Layer(int num_nodes, int num_inputs_per_node, ActivationFunction activation_function,
            const std::string& name, NodeType type);

        // Method to activate all nodes in the layer
        void activate(const std::vector<double>& inputs);

        // Method to backpropagate errors through the layer
        void backpropagate(const std::vector<double>& targets, double learning_rate, int saturation_threshold);

        // Method to print parameters of all nodes in the layer
        void print_parameters() const;

        // Accessor for the nodes
        const std::vector<Node>& get_nodes() const;
    };

} // namespace nn
