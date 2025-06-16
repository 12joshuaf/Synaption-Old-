#include "Net.hpp"
#include <string>

namespace nn {

	Net::~Net() {
		for (Layer* layer : this->layers) {
			delete layer;
		}
	}


	void Net::add_layer(int numNodes, int inputsPerNode, ActivationFunction activationType, NodeType type) {
		std::string name = "Node" + std::to_string(layers.size());
		Layer* addLayer = new Layer(numNodes, inputsPerNode, activationType, name, type);
		this->layers.push_back(addLayer);


		if (this->layers.size() > 1) {
			this->layers[this->layers.size() - 1]->connect_nodes(this->layers[this->layers.size()]);
			std::cout << "Layer added to net";
		}

	}


};
