#include "Perceptron.h"
#include "NeuronLayerMat.h"

namespace DirectGraphicalModels {
	namespace dnn {
		void CPerceptron::backPropagate(CNeuronLayerMat& layerA, CNeuronLayerMat& layerB, CNeuronLayerMat& layerC, const Mat& resultErrorRate, float learningRate)
		{
			Mat DeltaIn_j; // = layerC.getWeights() x resultErrorRate;
			gemm(layerC.getWeights(), resultErrorRate, 1, Mat(), 0, DeltaIn_j);

			// layerC.m_weights += learningRate * layerB.m_values x resultErrorRate.t()
			gemm(layerB.getValues(), resultErrorRate.t(), learningRate, layerC.getWeights(), 1, layerC.getWeights());

			//layerB.m_weights += learningRate * layerA.m_values x DeltaJ.t();
			gemm(layerA.getValues(), DeltaIn_j.t(), learningRate, layerB.getWeights(), 1, layerB.getWeights());
		}
	}
}