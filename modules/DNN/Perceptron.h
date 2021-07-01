#pragma once

#include "types.h"
#include "NeuronLayer.h"

namespace DirectGraphicalModels {
	namespace dnn {
		class CPerceptron {
		public:
			DllExport CPerceptron(const std::vector<int>& vNumNeurons);
			DllExport CPerceptron(const std::vector<ptr_nl_t>& vpLayers);
			DllExport CPerceptron(const CPerceptron&) = delete;
			DllExport ~CPerceptron(void) = default;
			
			DllExport bool operator=(const CPerceptron&) = delete;

			DllExport Mat	getPrediction(const Mat& inputValues);
			DllExport void	backPropagate(const Mat& solution, const Mat& gt, float learningRate);
		

		private:
			std::vector<ptr_nl_t> m_vpNeuronLayers;
		
		};
	}
}
