#include "Tensor.hpp"
#include <Eigen/Dense>
#include <iostream>

namespace nn {

    Tensor::Tensor(const std::vector<std::vector<double>>& inputs,
        const std::vector<std::vector<double>>& labels)
        : inputs(inputs), labels(labels) {}

    void Tensor::printInputs() const {
        std::cout << "Inputs:\n";
        for (const auto& row : inputs) {
            for (double val : row) {
                std::cout << val << " ";
            }
            std::cout << "\n";
        }
    }

    void Tensor::printLabels() const {
        std::cout << "Labels:\n";
        for (const auto& row : labels) {
            for (double val : row) {
                std::cout << val << " ";
            }
            std::cout << "\n";
        }
    }

    size_t Tensor::getNumSamples() const {
        return inputs.size();
    }

    size_t Tensor::getNumFeatures() const {
        return inputs.empty() ? 0 : inputs[0].size();
    }

    void performPCA(const std::vector<std::vector<double>>& input_data, int num_components) {
        if (input_data.empty()) {
            std::cerr << "Input data is empty.\n";
            return;
        }

        size_t n_samples = input_data.size();
        size_t n_features = input_data[0].size();

        Eigen::MatrixXd X(n_samples, n_features);
        for (size_t i = 0; i < n_samples; ++i) {
            if (input_data[i].size() != n_features) {
                std::cerr << "Inconsistent number of features in input data.\n";
                return;
            }
            for (size_t j = 0; j < n_features; ++j) {
                X(i, j) = input_data[i][j];
            }
        }

        // Center the data
        Eigen::RowVectorXd mean = X.colwise().mean();
        Eigen::MatrixXd centered = X.rowwise() - mean;

        // Covariance matrix
        Eigen::MatrixXd cov = (centered.transpose() * centered) / double(n_samples - 1);

        // Eigen decomposition
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(cov);
        if (eig.info() != Eigen::Success) {
            std::cerr << "Eigen decomposition failed.\n";
            return;
        }

        // Reverse order of eigenvectors (descending)
        Eigen::VectorXd eigenvalues = eig.eigenvalues().reverse();
        Eigen::MatrixXd eigenvectors = eig.eigenvectors().rowwise().reverse();

        // Project data
        Eigen::MatrixXd projected = centered * eigenvectors.leftCols(num_components);

        std::cout << "Projected Data (Top " << num_components << " Components):\n";
        std::cout << projected << std::endl;
    }

}
