#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <cmath>
#include <stdexcept>
#include <iomanip>

// Activation function enum
enum class ActivationFunction {
    Sigmoid,
    ReLU,
    Tanh,
    LeakyReLU,
    Step
};


//node type enum
enum class NodeType {
    Hidden,
    Output
};

// Activation functions and their derivatives
double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double sigmoid_derivative(double x) {
    const double sx = sigmoid(x);
    return sx * (1.0 - sx);
}

double relu(double x) {
    return x > 0 ? x : 0.0;
}

double relu_derivative(double x) {
    return x > 0 ? 1.0 : 0.0;
}

double tanh_activation(double x) {
    return std::tanh(x);
}

double tanh_derivative(double x) {
    double th = std::tanh(x);
    return 1.0 - th * th;
}

double leaky_relu(double x) {
    return x > 0 ? x : 0.01 * x;
}

double leaky_relu_derivative(double x) {
    return x > 0 ? 1.0 : 0.01;
}

double step(double x) {
    return x > 0 ? 1.0 : 0.0;
}

void warn_step_derivative() {
    std::cerr << "Warning: Step function has no derivative, only useful in perceptrons.\n";
}

// Node class
class Node {
private:
    ActivationFunction activation;//type of activation
    NodeType node_type;//type of node

    std::string layer_name;//name of layer
    std::string node_name;//name of node

    int saturation_count = 0;//member helps check for saturation in safe_backpropogation()


    //helps bakcpropogation
    double last_input_sum = 0.0;
    double last_delta = 0.0;
    double last_output = 0.0;


    //applies activation type to activaton()
    double apply_activation(double x) const {
        switch (activation) {
        case ActivationFunction::Sigmoid:
            return sigmoid(x);
        case ActivationFunction::ReLU:
            return relu(x);
        case ActivationFunction::Tanh:
            return tanh_activation(x);
        case ActivationFunction::LeakyReLU:
            return leaky_relu(x);
        case ActivationFunction::Step:
            return step(x);
        default:
            throw std::runtime_error("Unknown activation function!");
        }
    }

    //applies derrivstive of activation for backpropogation
    double activation_derivative(double x) const {
        switch (activation) {
        case ActivationFunction::Sigmoid:
            return sigmoid_derivative(x);
        case ActivationFunction::ReLU:
            return relu_derivative(x);
        case ActivationFunction::Tanh:
            return tanh_derivative(x);
        case ActivationFunction::LeakyReLU:
            return leaky_relu_derivative(x);
        case ActivationFunction::Step:
            std::cerr << "Warning: Step function has no defined derivative, only useful for perceptrons\n";
            return 0.0;
        default:
            throw std::runtime_error("Unknown activation function (derivative)!");
        }
    }

public:
    std::vector<double> weights;//weights for inputs
    double bias;//bias

    //constructor randomly assigns values to weights and biases before training begins
    Node(int num_inputs, ActivationFunction activation_input, const std::string& layer,
        const std::string& name, NodeType type)
        : activation(activation_input), layer_name(layer), node_name(name), node_type(type)
    {
        std::random_device rd;
        std::mt19937 eng(rd());

        if (activation == ActivationFunction::ReLU || activation == ActivationFunction::LeakyReLU) {
            std::normal_distribution<> he_dist(0.0, std::sqrt(2.0 / num_inputs));
            weights.resize(num_inputs);
            for (int i = 0; i < num_inputs; ++i) {
                weights[i] = he_dist(eng);
            }
            bias = he_dist(eng);
        }
        else {
            std::uniform_real_distribution<> distr(-1.0, 1.0);
            weights.resize(num_inputs);
            for (int i = 0; i < num_inputs; ++i) {
                weights[i] = distr(eng);
            }
            bias = distr(eng);
        }
    }

    //destructor
    ~Node() {
        std::cout << "Destroying Node: " << node_name << " in " << layer_name << "\n";
    }


    //functions for backpropogation
    double get_last_delta() const {
        return last_delta;
    }

    double get_last_output() const {
        return last_output;
    }

    std::vector<double> get_weights() const {
        return weights;
    }


    //activates forward
    double activate(const std::vector<double>& inputs) {
        if (inputs.size() != weights.size()) {
            throw std::invalid_argument("Input size does not match number of weights for node: " + node_name);
        }

        double sum = 0.0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            sum += inputs[i] * weights[i];
        }
        sum += bias;

        last_input_sum = sum;
        last_output = apply_activation(sum);
        return last_output;
    }


    //training functions
    void backpropagate(double error, double learning_rate, const std::vector<double>& inputs) {
        double derivative = activation_derivative(last_input_sum);
        double delta = error * derivative;
        last_delta = delta;

        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] -= learning_rate * delta * inputs[i];
        }
        bias -= learning_rate * delta;
    }


    //backpropogation with checks for saturation
    void safe_backpropagate(double error, double learning_rate, const std::vector<double>& inputs) {
        double previous_bias = bias;
        std::vector<double> previous_weights = weights;

        double derivative = activation_derivative(last_input_sum);
        double delta = error * derivative;
        last_delta = delta;

        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] -= learning_rate * delta * inputs[i];
        }
        bias -= learning_rate * delta;

        bool unchanged = true;
        const double epsilon = 1e-9;

        for (size_t i = 0; i < weights.size(); ++i) {
            if (std::abs(weights[i] - previous_weights[i]) > epsilon) {
                unchanged = false;
                break;
            }
        }
        if (std::abs(bias - previous_bias) > epsilon) {
            unchanged = false;
        }

        if (unchanged) {
            saturation_count++;
            if (saturation_count > 9) {
                std::cout << "Warning! Possible Saturation\n";
                print_parameters();
            }
        }
    }

    //prints the name, layer, weights and bias of the node
    void print_parameters() const {
        std::cout << std::fixed << std::setprecision(10);
        std::cout << "Node: " << node_name << " in " << layer_name << "\nWeights: ";
        for (double w : weights) {
            std::cout << w << " ";
        }
        std::cout << "\nBias: " << bias << "\n";
    }
};

// example usage
int main() {
    Node node(2, ActivationFunction::ReLU, "Layer 1", "Node0", NodeType::Hidden);

    std::vector<double> inputs = { 1.0, 1.0 };
    double output = node.activate(inputs);

    std::cout << "Node output: " << output << std::endl;

    double dummy_error = 0.5;
    double learning_rate = 0.1;
    node.print_parameters();


    node.safe_backpropagate(dummy_error, learning_rate, inputs);

    std::cout << "\nAfter backpropagation:\n";
    node.print_parameters();

    std::cout << "\nPress enter to close\n";
    std::cin.get();

    return 0;
}
