#pragma once

#include "Layer.hpp"
#include "Tensor.hpp"
#include <vector>
#include <string>
#include <stdexcept>


namespace nn {

	class Net {

	public:
		~Net();


		void add_layer(int numNodes, int inputsPerNode, ActivationFunction activationType, NodeType type);

	private:
		std::vector<Layer*> layers;


	};

};


