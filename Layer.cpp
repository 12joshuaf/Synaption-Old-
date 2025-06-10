#include "Layer.hpp"

namespace nn {

    // Constructor to create a layer with a specified number of nodes
    Layer::Layer(int num_nodes, int num_inputs_per_node, ActivationFunction activation_function,
        const std::string& name, NodeType type)
        : layer_name(name)
    {
        nodes.reserve(num_nodes);
        for (int i = 0; i < num_nodes; ++i) {
            nodes.emplace_back(num_inputs_per_node, activation_function, layer_name,
                "Node_" + std::to_string(i), type);
        }
    }

    // Method to activate all nodes in the layer
    void Layer::activate(const std::vector<double>& inputs) {
        for (Node& node : nodes) {
            node.activate(inputs);
        }
    }

    // Method to backpropagate errors through the layer
    void Layer::backpropagate(const std::vector<double>& targets, double learning_rate, int saturation_threshold) {
        for (size_t i = 0; i < nodes.size(); ++i) {
            if (nodes[i].get_last_output() != 0) { // Only backpropagate for output nodes
                nodes[i].backpropagate(targets[i], learning_rate, saturation_threshold);
            }
            else {
                nodes[i].backpropagate(learning_rate, saturation_threshold);
            }
        }
    }

    // Method to print parameters of all nodes in the layer
    void Layer::print_parameters() const {
        for (const Node& node : nodes) {
            node.print_parameters();
        }
    }

    // Accessor for the nodes
    const std::vector<Node>& Layer::get_nodes() const {
        return nodes;
    }

} // namespace nn
