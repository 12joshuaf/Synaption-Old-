#include "Net.hpp"
#include <string>
#include <fstream>
#include <iostream>

namespace nn {

	Net::Net() {
		numLayers = 0;
	}

	Net::~Net() {
		for (Layer* layer : layers) {
			delete layer;
		}
	}

	void Net::add_layer(int numNodes, int inputsPerNode, ActivationFunction activationType, NodeType type) {
		std::string name = "Node" + std::to_string(layers.size());
		Layer* addLayer = new Layer(numNodes, inputsPerNode, activationType, name, type);

		if (!layers.empty()) {
			layers.back()->connect_nodes(addLayer);
		}
		layers.push_back(addLayer);
		numLayers++;
	}

	void Net::print_parameters(bool verbose) const {
		std::cout << "Network parameters:\n";
		for (size_t i = 0; i < layers.size(); ++i) {
			std::cout << "Layer " << i << " (" << layers[i]->get_nodes().size() << " nodes):\n";
			layers[i]->print_parameters(verbose);
			std::cout << "\n";
		}
	}

	void Net::save_net(const std::string& fileName) const {
		std::ofstream outFile(fileName + ".snn");

		if (outFile.is_open()) {
			for (size_t layerIndex = 0; layerIndex < layers.size(); ++layerIndex) {
				const Layer* layer = layers[layerIndex];

				for (size_t nodeIndex = 0; nodeIndex < layer->get_nodes().size(); ++nodeIndex) {
					const Node& node = layer->get_nodes()[nodeIndex];

					outFile << "(";
					outFile << node.get_node_name() << ", ";
					outFile << "Layer" << layerIndex << ", ";

					switch (node.get_activation_function()) {
					case ActivationFunction::Sigmoid: outFile << "Sigmoid"; break;
					case ActivationFunction::ReLU: outFile << "ReLU"; break;
					case ActivationFunction::Tanh: outFile << "Tanh"; break;
					case ActivationFunction::LeakyReLU: outFile << "LeakyReLU"; break;
					case ActivationFunction::Step: outFile << "Step"; break;
					default: outFile << "Unknown"; break;
					}

					outFile << ", " << node.bias;

					for (double weight : node.get_weights()) {
						outFile << ", " << weight;
					}

					outFile << ")";
					if (nodeIndex != layer->get_nodes().size() - 1) {
						outFile << " ";
					}
				}
				outFile << "\n";
			}

			outFile.close();
			std::cout << "Network saved to " << fileName << ".snn\n";
		}
		else {
			std::cerr << "Error opening Net File!" << std::endl;
		}
	}

}
