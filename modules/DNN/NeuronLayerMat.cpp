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
	}
	
	void CNeuronLayerMat::generateRandomWeights(void)
	{
		m_weights = random::U(m_weights.size(), m_weights.type(), -0.5f, 0.5f);
	}


	void CNeuronLayerMat::dotProd(const CNeuronLayerMat& layer)
	{
		for (int y = 0; y < m_values.rows; y++) {
			float value = 0;
			for (int x = 0; x < layer.m_values.rows; x++)
				value += layer.m_weights.at<float>(x, y) * layer.m_values.at<float>(x, 0);
			m_values.at<float>(y, 0) = sigmoidFunction(value);
		}
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
		Mat DeltaWjk(layerB.getNumNeurons(), layerC.getNumNeurons(), CV_32FC1);
		Mat DeltaJ(layerB.getNumNeurons(), 1, CV_32FC1);

		for (int i = 0; i < layerB.getNumNeurons(); i++) {
			float nodeVal = 0;
			for (int j = 0; j < layerC.getNumNeurons(); j++) {
				nodeVal += layerB.m_weights.at<float>(i, j) * resultErrorRate.at<float>(j, 0);
				DeltaWjk.at<float>(i, j) = learningRate * resultErrorRate.at<float>(j, 0) * layerB.m_values.at<float>(i, 0);
			}
			float sigmoid = sigmoidFunction(layerB.m_values.at<float>(i, 0));
			DeltaJ.at<float>(i, 0) = nodeVal * sigmoid * (1 - sigmoid);
		}

		for (int i = 0; i < layerA.getNumNeurons(); i++) 
			for (int j = 0; j < layerB.getNumNeurons(); j++) 
				layerA.m_weights.at<float>(i, j) += learningRate * DeltaJ.at<float>(j, 0) * layerA.m_values.at<float>(i, 0);;

		layerB.m_weights += DeltaWjk;
	}
}}
