#pragma once

#include "Node.hpp"
#include <vector>
#include <string>
#include <stdexcept>

namespace nn {

    class Layer {
    public:
        Layer(int num_nodes, int num_inputs_per_node, ActivationFunction activation_function,
            const std::string& name = "", NodeType type = NodeType::Hidden);

        void activate(const std::vector<double>& inputs);


        void backpropagate(const std::vector<double>& targets, double learning_rate, int saturation_threshold);


        void backpropagate(double learning_rate, int saturation_threshold);

        void print_parameters(bool verbose = true) const;

        void connect_nodes(Layer* layer);

        const std::vector<Node>& get_nodes() const;

        std::vector<double> get_outputs() const;

        void add_node(Node);

        std::vector<Node> nodes;

        NodeType layerType;


    private:
        std::string layer_name;
    };

} // namespace nn
