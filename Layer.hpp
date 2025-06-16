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

        // Overload for output layer with targets
        void backpropagate(const std::vector<double>& targets, double learning_rate, int saturation_threshold);

        // Overload for hidden layer without targets
        void backpropagate(double learning_rate, int saturation_threshold);

        void print_parameters(bool verbose = true) const;

        const std::vector<Node>& get_nodes() const;

        std::vector<double> get_outputs() const;

    private:
        std::vector<Node> nodes;
        std::string layer_name;
    };

} // namespace nn
