#pragma once

#include "Layer.hpp"
#include "Tensor.hpp"
#include <vector>
#include <string>
#include <stdexcept>
#include <fstream>


namespace nn {

	class Net {

	public:
		~Net();

		Net();

		bool isEmpty;
		int numLayers;


		void add_layer(int numNodes, int inputsPerNode, ActivationFunction activationType, NodeType type);

		Layer* get_layer(size_t index) {
			if (index >= layers.size()) throw std::out_of_range("Invalid layer index");
			return layers[index];
		}

		void print_parameters(bool verbose = true) const;


		std::vector<Layer*> layers;


		void save_net(const std::string& fileName) const;

		void load_net(const std::string& fileName);

		void activate(const std::vector<double>& inputs);
		void backpropagate(const std::vector<double>& targets, double learning_rate, int saturation_threshold);



	};

};

