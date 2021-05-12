#pragma once

#include "types.h"

namespace DirectGraphicalModels {
	namespace dnn {
		class CNeuronLayer;

		class CPerceptron {
		public:
			DllExport static void backPropagate(CNeuronLayer& layerA, CNeuronLayer& layerB, CNeuronLayer& layerC, const Mat& resultErrorRate, float learningRate);
		};
	}
}
