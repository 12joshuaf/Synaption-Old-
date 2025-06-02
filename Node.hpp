#ifndef NODE_HPP
#define NODE_HPP

#include <vector>
#include <string>
#include <stdexcept>
#include <random>
#include <iostream>
#include <iomanip>
#include <cmath>

// Activation function enums
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

// Activation function declarations
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

// Node class declaration
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

    double apply_activation(double x) const;
    double activation_derivative(double x) const;

public:
    std::vector<double> weights;
    double bias;


    //constructor
    Node(int num_inputs, ActivationFunction activation_input, const std::string& layer,
        const std::string& name, NodeType type);
    ~Node();


    //functions for bakcpropogation
    double get_last_delta() const;
    double get_last_output() const;
    std::vector<double> get_weights() const;


    //activating node
    double activate(const std::vector<double>& inputs);

    //backpropogation
    void backpropagate(double error, double learning_rate, const std::vector<double>& inputs);

    //backpropogation that checks for saturation
    void safe_backpropagate(double error, double learning_rate, const std::vector<double>& inputs);

    //prints the name, later name, weights and bias of the node
    void print_parameters() const;
};

#endif // NODE_HPP
