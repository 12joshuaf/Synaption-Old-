#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <cmath>
#include <stdexcept>
#include <iomanip>

namespace nn {
    //activation enums
    enum class ActivationFunction {
        Sigmoid,
        ReLU,
        Tanh,
        LeakyReLU,
        Step
    };
    //node types for building neural nets
    enum class NodeType {
        Hidden,
        Output
    };

    // Activation functions
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
        ActivationFunction activation; //type of activation to be used from enum
        NodeType node_type; //hidde or output from enum

        std::string layer_name; //name of layer for debbuing nn
        std::string node_name; //name of node for debugging nn

        int saturation_count = 0; //member for detecting learning saturation


        //members for backpropogation
        double last_input_sum = 0.0;
        double last_delta = 0.0;
        double last_output = 0.0;


        //functions for activation and backpropogation using activation functions from enum
        double apply_activation(double x) const;
        double activation_derivative(double x) const;

    public:
        std::vector<double> weights; //weights for inputs w0 -> wn
        double bias; //bias for activation


        //constructor and destructor
        Node(int num_inputs, ActivationFunction activation_input, const std::string& layer,
            const std::string& name, NodeType type);
        ~Node();


        //functions for backpropogation
        double get_last_delta() const;
        double get_last_output() const;

        //returns weights for saving nn
        std::vector<double> get_weights() const;


        //functions for training
        double activate(const std::vector<double>& inputs);
        void backpropagate(double error, double learning_rate, const std::vector<double>& inputs);
        void safe_backpropagate(double error, double learning_rate, const std::vector<double>& inputs); //safe backpropogate checks for saturation but has extra time complexity
        void print_parameters() const;
    };

} // namespace nn


