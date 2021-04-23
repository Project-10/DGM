#include "NeuronLayer.h"
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
	
	
	
	// Constructor
	CNeuronLayer::CNeuronLayer(size_t numNeurons, size_t numConnection) 
	{
		for (size_t i = 0; i < numNeurons; i++)
			m_vpNeurons.push_back(std::make_shared<CNeuron>(numConnection));
	}

	void CNeuronLayer::generateRandomWeights(void) 
	{
		for(auto n : m_vpNeurons)
			n->generateRandomWeights();
	}

	void CNeuronLayer::setValues(const Mat& values) 
	{
		// Assertions
		DGM_ASSERT(values.type() == CV_32FC1);
		DGM_ASSERT(values.cols == 1);
		DGM_ASSERT(values.rows == m_vpNeurons.size());
		
		for (size_t i = 0; i < m_vpNeurons.size(); i++)
			m_vpNeurons[i]->setValue(values.at<float>(static_cast<int>(i), 0));
	}

	Mat CNeuronLayer::getValues(void) const
	{
		Mat res(m_vpNeurons.size(), 1, CV_32FC1);
		for (size_t i = 0; i < m_vpNeurons.size(); i++)
			res.at<float>(i, 0) = m_vpNeurons[i]->getValue();
		return res;
	}

	void CNeuronLayer::dotProd(const CNeuronLayer& layer) 
	{	
		for (size_t i = 0; i < m_vpNeurons.size(); i++) {
			float value = 0;
			for (size_t j = 0; j < layer.getNumNeurons(); j++)
				value += m_vpNeurons[i]->getWeight(j) * layer.m_vpNeurons[j]->getValue();
			value = sigmoidFunction(value);
			m_vpNeurons[i]->setValue(value);
		}
	}

	void CNeuronLayer::backPropagate(CNeuronLayer& layerA, CNeuronLayer& layerB, CNeuronLayer& layerC, const Mat& resultErrorRate, float learningRate)
	{
		Mat DeltaWjk(layerC.getNumNeurons(), layerB.getNumNeurons(), CV_32FC1);
		std::vector<float> DeltaJ(layerB.getNumNeurons());

		for (size_t j = 0; j < layerB.getNumNeurons(); j++) {
			float nodeVal = 0;
			for (size_t i = 0; i < layerC.getNumNeurons(); i++) {
				nodeVal += layerC.m_vpNeurons[i]->getWeight(j) * resultErrorRate.at<float>(i, 0);
				DeltaWjk.at<float>(i, j) = learningRate * resultErrorRate.at<float>(i, 0) * layerB.m_vpNeurons[j]->getValue();
			}
			float sigmoid = sigmoidFunction(layerB.m_vpNeurons[j]->getValue());
			DeltaJ[j] = nodeVal * sigmoid * ( 1 - sigmoid);
		}

		for (size_t i = 0; i < layerB.getNumNeurons(); i++) {
			for (size_t j = 0; j < layerA.getNumNeurons(); j++) {
				float Delta = learningRate * DeltaJ[i] * layerA.m_vpNeurons[j]->getValue();
				float oldWeight = layerB.m_vpNeurons[i]->getWeight(j);
				layerB.m_vpNeurons[i]->setWeight(j, oldWeight + Delta);
			}
		}

		for (size_t i = 0; i < layerC.getNumNeurons(); i++) {
			for (size_t j = 0; j < layerB.getNumNeurons(); j++) {
				float oldWeight = layerC.m_vpNeurons[i]->getWeight(j);
				layerC.m_vpNeurons[i]->setWeight(j, oldWeight + DeltaWjk.at<float>(i, j));
			}
		}
	}


}}
