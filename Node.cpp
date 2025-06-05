#include "Node.hpp"

namespace nn {

    // Activation functions
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

    // Constructor
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

    // Destructor
    Node::~Node() {
        std::cout << "Destroying Node: " << node_name << " in " << layer_name << "\n";
    }

    void Node::point_node(std::vector<Node*> node_vector) {
        this->points_to = node_vector;
        for (Node* n : node_vector) {
            n->inputs_from.push_back(this);
        }
    }

    void Node::point_node(Node* n) {
        this->points_to.push_back(n);
        n->inputs_from.push_back(this);
    }

    void Node::input_nodes(std::vector<Node*> node_vector) {
        this->inputs_from = node_vector;
    }

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
            warn_step_derivative();
            return 0.0;
        default:
            throw std::runtime_error("Unknown activation function (derivative)!");
        }
    }

    double Node::activate() {
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

        for (Node* n : points_to) {
            n->inputs.push_back(last_output);
        }

        return last_output;
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

        for (Node* n : points_to) {
            n->inputs.push_back(last_output);
        }

        return last_output;
    }

    // Safe backpropagation for output nodes
    void Node::backpropagate(double target, double learning_rate, int saturation_threshold) {
        if (node_type != NodeType::Output) {
            throw std::logic_error("safe_backpropagate(target) should only be called on output nodes.");
        }

        double error = target - last_output;
        last_delta = error * activation_derivative(last_input_sum);

        if (std::abs(last_delta) < 1e-6) {
            saturation_count++;
            if (saturation_count >= saturation_threshold) {
                std::cerr << "Warning: Node '" << node_name
                    << "' in layer '" << layer_name
                    << "' appears saturated (delta = 0 for "
                    << saturation_threshold << " steps).\n";
                saturation_count = 0;
            }
        }
        else {
            saturation_count = 0;
        }

        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] += learning_rate * last_delta * inputs[i];
        }
        bias += learning_rate * last_delta;

        for (size_t i = 0; i < inputs_from.size(); ++i) {
            inputs_from[i]->back_inputs.push_back(last_delta * weights[i]);
        }
    }

    // Safe backpropagation for hidden nodes
    void Node::backpropagate(double learning_rate, int saturation_threshold) {
        if (node_type != NodeType::Hidden) {
            throw std::logic_error("safe_backpropagate() should only be called on hidden nodes.");
        }

        double sum = 0.0;
        for (double val : back_inputs) {
            sum += val;
        }

        last_delta = sum * activation_derivative(last_input_sum);

        if (std::abs(last_delta) < 1e-6) {
            saturation_count++;
            if (saturation_count >= saturation_threshold) {
                std::cerr << "Warning: Node '" << node_name
                    << "' in layer '" << layer_name
                    << "' appears saturated (delta = 0 for "
                    << saturation_threshold << " steps).\n";
                saturation_count = 0;
            }
        }
        else {
            saturation_count = 0;
        }

        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] += learning_rate * last_delta * inputs[i];
        }
        bias += learning_rate * last_delta;

        for (size_t i = 0; i < inputs_from.size(); ++i) {
            inputs_from[i]->back_inputs.push_back(last_delta * weights[i]);
        }

        back_inputs.clear();
    }

    void Node::print_parameters() const {
        std::cout << std::fixed << std::setprecision(10);
        std::cout << "Node: " << node_name << " in " << layer_name << "\nWeights: ";
        for (double w : weights) {
            std::cout << w << " ";
        }
        std::cout << "\nBias: " << bias << "\n";
    }

} // namespace nn
