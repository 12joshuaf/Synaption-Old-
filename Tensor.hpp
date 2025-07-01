#pragma once

#include <vector>
#include <iostream>

namespace nn {

    class Tensor {
    public:
        std::vector<std::vector<double>> inputs;  // Each inner vector is a sample
        std::vector<std::vector<double>> labels;

        Tensor(const std::vector<std::vector<double>>& inputs,
            const std::vector<std::vector<double>>& labels);

        void printInputs() const;
        void printLabels() const;

        size_t getNumSamples() const;
        size_t getNumFeatures() const;
    };

    // PCA function
    void performPCA(const std::vector<std::vector<double>>& input_data, int num_components);

}
