#include "Tensor.hpp"

namespace tensor {

    Tensor::Tensor(const std::vector<double>& inputs, const std::vector<double>& labels)
        : inputs(inputs), labels(labels) {}

    void Tensor::printInputs() const {
        std::cout << "Inputs: ";
        for (double val : inputs) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    void Tensor::printLabels() const {
        std::cout << "Labels: ";
        for (double val : labels) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    size_t Tensor::getInputLength() const {
        return inputs.size();
    }

    size_t Tensor::getLabelLength() const {
        return labels.size();
    }

}
