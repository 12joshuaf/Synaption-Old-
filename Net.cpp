#include "Net.hpp"
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <tuple>



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


	void Net::load_net(const std::string& fileName) {
		if (!(fileName.size() >= 4 && fileName.substr(fileName.length() - 4) == ".snn")) {
			std::cerr << "Incorrect file suffix, should be .snn\n";
			return;
		}
		std::ifstream inputFile(fileName);
		if (!inputFile.is_open()) {
			std::cerr << "Error: Could not open file." << std::endl;
			return;
		}
s
		for (Layer* layer : layers) {
			delete layer;
		}
		layers.clear();
		numLayers = 0;

		std::string line;
		int layerIndex = 0;
		while (std::getline(inputFile, line)) {
			std::vector<std::tuple<std::string, ActivationFunction, double, std::vector<double>>> nodeDataList;


			size_t pos = 0;
			while ((pos = line.find('(')) != std::string::npos) {
				size_t end = line.find(')', pos);
				if (end == std::string::npos) break;

				std::string nodeStr = line.substr(pos + 1, end - pos - 1);
				line = line.substr(end + 1);

				std::istringstream ss(nodeStr);
				std::string token;


				std::getline(ss, token, ',');
				std::string nodeName = token;

				std::getline(ss, token, ',');

				std::getline(ss, token, ',');
				std::string actStr = token;
				actStr.erase(remove_if(actStr.begin(), actStr.end(), ::isspace), actStr.end());

				ActivationFunction af;
				if (actStr == "Sigmoid") af = ActivationFunction::Sigmoid;
				else if (actStr == "ReLU") af = ActivationFunction::ReLU;
				else if (actStr == "Tanh") af = ActivationFunction::Tanh;
				else if (actStr == "LeakyReLU") af = ActivationFunction::LeakyReLU;
				else if (actStr == "Step") af = ActivationFunction::Step;
				else af = ActivationFunction::Sigmoid; // fallback default

				// 4. Bias
				std::getline(ss, token, ',');
				double bias = std::stod(token);


				std::vector<double> weights;
				while (std::getline(ss, token, ',')) {
					weights.push_back(std::stod(token));
				}

				nodeDataList.emplace_back(nodeName, af, bias, weights);
			}

			// Build the layer
			if (!nodeDataList.empty()) {
				int numNodes = static_cast<int>(nodeDataList.size());
				size_t inputsPerNode = std::get<3>(nodeDataList[0]).size();
				bool isLastLayer = inputFile.peek() == EOF;

				NodeType type = isLastLayer ? NodeType::Output : NodeType::Hidden;

				Layer* newLayer = new Layer(
					numNodes,
					static_cast<int>(inputsPerNode),
					std::get<1>(nodeDataList[0]),
					"Layer" + std::to_string(layerIndex),
					type
				);


				for (int i = 0; i < numNodes; ++i) {
					Node& node = const_cast<Node&>(newLayer->get_nodes()[i]);

					node.get_weights() = std::get<3>(nodeDataList[i]);
					node.get_node_name() = std::get<0>(nodeDataList[i]);
					node.bias = std::get<2>(nodeDataList[i]);
				}

				// Connect previous layer to this one
				if (!layers.empty()) {
					layers.back()->connect_nodes(newLayer);
				}

				layers.push_back(newLayer);
				layerIndex++;
			}
		}

		numLayers = static_cast<int>(layers.size());
		std::cout << "Network loaded from " << fileName << "\n";
	}




}
