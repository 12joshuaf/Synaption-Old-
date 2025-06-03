#pragma once

#include <vector>
#include <iostream>

namespace tensor {

    class Tensor {
    public:
        std::vector<double> inputs;
        std::vector<double> labels;

        Tensor(const std::vector<double>& inputs, const std::vector<double>& labels);

        void printInputs() const;
        void printLabels() const;

        size_t getInputLength() const;
        size_t getLabelLength() const;
    };

}


