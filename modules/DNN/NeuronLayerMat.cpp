#include "NeuronLayerMat.h"
#include "DGM/random.h"
#include "DGM/parallel.h"
#include "macroses.h"

namespace DirectGraphicalModels { namespace dnn
{
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
	
		void sigmoidFunction(Mat& X)
		{
			for (int y = 0; y < X.rows; y++){
				float* pX = X.ptr<float>(y);
				for (int x = 0; x < X.cols; x++)
					pX[x] = sigmoidFunction(pX[x]);
			}
		}
	}
	
	void CNeuronLayerMat::generateRandomWeights(void)
	{
		m_weights = random::U(m_weights.size(), m_weights.type(), -0.5f, 0.5f);
	}


	void CNeuronLayerMat::dotProd(const CNeuronLayerMat& layer)
	{
		// this->m_values = this->m_weights * layer.m_values;
		parallel::gemm(m_weights.t(), layer.m_values, 1, Mat(), 0, m_values);
		sigmoidFunction(m_values);
	}

	void CNeuronLayerMat::setValues(const Mat& values)
	{
		// Assertions
		DGM_ASSERT(values.type() == m_values.type());
		DGM_ASSERT(values.size() == m_values.size());
		values.copyTo(m_values);
	}

	void CNeuronLayerMat::backPropagate(CNeuronLayerMat& layerA, CNeuronLayerMat& layerB, CNeuronLayerMat& layerC, const Mat& resultErrorRate, float learningRate)
	{
		Mat DeltaIn_j; // = layerB.getWeights() x resultErrorRate;
		parallel::gemm(layerC.m_weights, resultErrorRate, 1, Mat(), 0, DeltaIn_j);

		Mat DeltaJ(layerB.getNumNeurons(), 1, CV_32FC1); // 60 x 1
		for(int i=0; i < layerB.getNumNeurons(); i++){
			DeltaJ.at<float>(i,0) = DeltaIn_j.at<float>(i,0) * sigmoidFunction_derivative(layerB.m_values.at<float>(i, 0));
		}
		
//		Mat DeltaJ(layerB.getNumNeurons(), 1, CV_32FC1);
//		for (int i = 0; i < layerB.getNumNeurons(); i++) {
//			float nodeVal = 0;
//			for (int j = 0; j < layerC.getNumNeurons(); j++)
//				nodeVal += layerC.m_weights.at<float>(i, j) * resultErrorRate.at<float>(j, 0);
//
//			float sigmoid = sigmoidFunction(layerB.m_values.at<float>(i, 0));
//			DeltaJ.at<float>(i, 0) = nodeVal * sigmoidFunction_derivative(sigmoid);
//		}

		// layerC.m_weights += learningRate * layerB.m_values x resultErrorRate.t()
		parallel::gemm(layerB.m_values, resultErrorRate.t(), learningRate, layerC.m_weights, 1, layerC.m_weights);
		
		//layerB.m_weights += learningRate * layerA.m_values x DeltaJ.t();
		parallel::gemm(layerA.m_values, DeltaJ.t(), learningRate, layerB.m_weights, 1, layerB.m_weights);

	}
}}
