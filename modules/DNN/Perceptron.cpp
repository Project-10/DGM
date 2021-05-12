#include "Perceptron.h"
#include "NeuronLayer.h"

namespace DirectGraphicalModels {
	namespace dnn {


		namespace {
			/**
			 * Applies the Sigmoid Activation function
			 *
			 * @param the value at each node
			 * @return a number between 0 and 1.
			 */
			float sigmoidFunction(float x)
			{
				return 1.0f / (1.0f + expf(-x));
			}

			float sigmoidFunction_derivative(float x)
			{
				float s = sigmoidFunction(x);
				return s * (1 - s);
			}
		}

		void CPerceptron::backPropagate(CNeuronLayer& layerA, CNeuronLayer& layerB, CNeuronLayer& layerC, const Mat& solution, const Mat& gt, float learningRate)
		{
			Mat error = gt - solution;
			for (int i = 0; i < error.rows; i++)
				error.at<float>(i, 0) *= sigmoidFunction_derivative(solution.at<float>(i, 0));
			
			Mat DeltaIn_j; // = layerC.getWeights() x resultErrorRate;
			gemm(layerC.getWeights(), error, 1, Mat(), 0, DeltaIn_j);

			// layerC.m_weights += learningRate * layerB.m_values x resultErrorRate.t()
			gemm(layerB.getValues(), error.t(), learningRate, layerC.getWeights(), 1, layerC.getWeights());

			//layerB.m_weights += learningRate * layerA.m_values x DeltaJ.t();
			gemm(layerA.getValues(), DeltaIn_j.t(), learningRate, layerB.getWeights(), 1, layerB.getWeights());
		}
	}
}