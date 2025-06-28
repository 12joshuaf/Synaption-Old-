#include "Layer.hpp"
#include <iostream> 
#include <stdexcept>

namespace nn {

    Layer::Layer(int num_nodes, int num_inputs_per_node, ActivationFunction activation_function,
        const std::string& name, NodeType type)
        : layer_name(name)
    {
        nodes.reserve(num_nodes);
        for (int i = 0; i < num_nodes; ++i) {
            nodes.emplace_back(num_inputs_per_node, activation_function, layer_name,
                "Node_" + std::to_string(i), type);
        }

        this->layerType = type;
    }

    void Layer::activate(const std::vector<double>& inputs) {
        for (Node& node : nodes) {
            node.activate(inputs);
        }
    }

    void Layer::backpropagate(const std::vector<double>& targets, double learning_rate, int saturation_threshold) {
        if (targets.size() != nodes.size()) {
            std::cerr << "Error: Target size = " << targets.size()
                << ", expected = " << nodes.size() << std::endl;
            throw std::invalid_argument("Target size does not match the number of nodes in the layer.");
        }

        for (size_t i = 0; i < nodes.size(); ++i) {
            if (nodes[i].get_node_type() == NodeType::Output) {
                nodes[i].backpropagate(targets[i], learning_rate, saturation_threshold);
            }
            else {
                nodes[i].backpropagate(learning_rate, saturation_threshold);
            }
        }
    }

    void Layer::backpropagate(double learning_rate, int saturation_threshold) {
        for (Node& node : nodes) {
            node.backpropagate(learning_rate, saturation_threshold);
        }
    }

    void Layer::print_parameters(bool verbose) const {
        for (const Node& node : nodes) {
            if (verbose) {
                node.print_parameters();
            }
        }
    }

    const std::vector<Node>& Layer::get_nodes() const {
        return nodes;
    }

    std::vector<double> Layer::get_outputs() const {
        std::vector<double> outputs;
        outputs.reserve(nodes.size());
        for (const Node& node : nodes) {
            outputs.push_back(node.get_last_output());
        }
        return outputs;
    }

    void Layer::connect_nodes(Layer* next_layer) {
        for (Node& source_node : this->nodes) {
            for (Node& target_node : next_layer->nodes) {
                source_node.point_node(&target_node);
            }
        }
    }


    void Layer::add_node(Node node) {
        this->nodes.push_back(node);
    }



} 
