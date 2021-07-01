#pragma once

#include "types.h"
#include "NeuronLayer.h"

namespace DirectGraphicalModels {
	namespace dnn {
		class CPerceptron {
		public:
			DllExport CPerceptron(const std::vector<int>& vNumNeurons);
			DllExport CPerceptron(const std::vector<ptr_nl_t>& vpLayers); // added numHiddenLayer for passing
			DllExport CPerceptron(const CPerceptron&) = delete;                                                        // vector of hidden layers
			DllExport ~CPerceptron(void) = default;
			
			DllExport bool operator=(const CPerceptron&) = delete;

			DllExport Mat	getPrediction(const Mat& inputValues);
			DllExport void	backPropagate(const Mat& solution, const Mat& gt, float learningRate, int numHiddenLayer);
		

		private:
			std::vector<ptr_nl_t> m_vpNeuronLayers;
		
		};
	}
}
