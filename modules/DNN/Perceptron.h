#pragma once

#include "types.h"

namespace DirectGraphicalModels {
	namespace dnn {
		class CNeuronLayerMat;

		class CPerceptron {
		public:
			DllExport static void backPropagate(CNeuronLayerMat& layerA, CNeuronLayerMat& layerB, CNeuronLayerMat& layerC, const Mat& resultErrorRate, float learningRate);
		};
	}
}
