#include "Perceptron.h"
#include "NeuronLayer.h"

namespace DirectGraphicalModels {
	namespace dnn {
		void CPerceptron::backPropagate(CNeuronLayer& layerA, CNeuronLayer& layerB, CNeuronLayer& layerC, const Mat& solution, const Mat& gt, float learningRate)
		{
			Mat error = gt - solution;
			for (int i = 0; i < error.rows; i++)
				error.at<float>(i, 0) *= layerC.getActivationFunctionDeriateve()(solution.at<float>(i, 0));
			
			Mat DeltaIn_j; // = layerC.getWeights() x resultErrorRate;
			gemm(layerC.getWeights(), error, 1, Mat(), 0, DeltaIn_j);

			// layerC.m_weights += learningRate * layerB.m_values x resultErrorRate.t()
			gemm(layerB.getValues(), error.t(), learningRate, layerC.getWeights(), 1, layerC.getWeights());

			//layerB.m_weights += learningRate * layerA.m_values x DeltaJ.t();
			gemm(layerA.getValues(), DeltaIn_j.t(), learningRate, layerB.getWeights(), 1, layerB.getWeights());
		}
	}
}