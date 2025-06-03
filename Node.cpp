#include "Node.hpp"

namespace nn {


    //functions for activation
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

    //step function hss no derrivtive, but perceptrons are supported. If step function is used in neural net, the user will be warned
    void warn_step_derivative() {
        std::cerr << "Warning: Step function has no derivative, only useful in perceptrons.\n";
    }


    //constructor
    Node::Node(int num_inputs, ActivationFunction activation_input, const std::string& layer,
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
    Node::~Node() {
        std::cout << "Destroying Node: " << node_name << " in " << layer_name << "\n";
    }


    //functions for backpropogation and activaton
    double Node::get_last_delta() const {
        return last_delta;
    }

    double Node::get_last_output() const {
        return last_output;
    }

    std::vector<double> Node::get_weights() const {
        return weights;
    }

    double Node::apply_activation(double x) const {
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

    double Node::activation_derivative(double x) const {
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


    double Node::activate(const std::vector<double>& inputs) {
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

    void Node::backpropagate(double error, double learning_rate, const std::vector<double>& inputs) {
        double derivative = activation_derivative(last_input_sum);
        double delta = error * derivative;
        last_delta = delta;

        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] -= learning_rate * delta * inputs[i];
        }
        bias -= learning_rate * delta;
    }


    //safe backpropogation checks for saturation, but has extra time complexity
    void Node::safe_backpropagate(double error, double learning_rate, const std::vector<double>& inputs) {
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
                saturation_count = 0;
                print_parameters();
            }
        }
    }

    //for debugging
    void Node::print_parameters() const {
        std::cout << std::fixed << std::setprecision(10);
        std::cout << "Node: " << node_name << " in " << layer_name << "\nWeights: ";
        for (double w : weights) {
            std::cout << w << " ";
        }
        std::cout << "\nBias: " << bias << "\n";
    }

} // namespace nn
