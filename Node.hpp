#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <cmath>
#include <stdexcept>
#include <iomanip>

namespace nn {

    enum class ActivationFunction {
        Sigmoid,
        ReLU,
        Tanh,
        LeakyReLU,
        Step
    };

    enum class NodeType {
        Hidden,
        Output
    };

    double sigmoid(double x);
    double sigmoid_derivative(double x);
    double relu(double x);
    double relu_derivative(double x);
    double tanh_activation(double x);
    double tanh_derivative(double x);
    double leaky_relu(double x);
    double leaky_relu_derivative(double x);
    double step(double x);
    void warn_step_derivative();

    class Node {
    private:
        ActivationFunction activation;
        NodeType node_type;

        std::string layer_name;
        std::string node_name;

        int saturation_count = 0;

        double last_input_sum = 0.0;
        double last_delta = 0.0;
        double last_output = 0.0;

        std::vector<double> inputs_snapshot;

        double apply_activation(double x) const;
        double activation_derivative(double x) const;

    public:
        std::vector<double> weights;
        double bias;

        std::vector<Node*> points_to;
        std::vector<Node*> inputs_from;
        std::vector<double> inputs;
        std::vector<double> back_inputs;

        Node(int num_inputs, ActivationFunction activation_input, const std::string& layer,
            const std::string& name, NodeType type);
        ~Node();

        void point_node(std::vector<Node*> node_vector);
        void point_node(Node* node);
        void input_nodes(std::vector<Node*> node_vector);

        double get_last_delta() const;
        double get_last_output() const;
        std::vector<double> get_weights() const;

        double activate();
        double activate(const std::vector<double>& inputs);
        void print_parameters() const;

        void backpropagate(double target, double learning_rate, int saturation_threshold);
        void backpropagate(double learning_rate, int saturation_threshold);

        NodeType get_node_type();

        ActivationFunction get_activation_function() const;
        std::string get_node_name() const;

        std::vector<double>& get_weights();
        std::string& get_node_name();    
        void set_bias(double b);                 



    };

}
