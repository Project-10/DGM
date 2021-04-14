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
			
			for (const auto& n : layer.m_vpNeurons)
				value += n->getWeight(i) * n->getValue();

			value = sigmoidFunction(value);
			
			m_vpNeurons[i]->setValue(value);
		}
	}

	void CNeuronLayer::backPropagate(CNeuronLayer& layerA, CNeuronLayer& layerB, CNeuronLayer& layerC, std::vector<float>& vResultErrorRate, float learningRate)
	{
		Mat DeltaWjk(layerB.getNumNeurons(), layerC.getNumNeurons(), CV_32FC1);
		std::vector<float> DeltaJ(layerB.getNumNeurons());

		for (size_t i = 0; i < layerB.getNumNeurons(); i++) {
			float nodeVal = 0;
			for (size_t j = 0; j < layerC.getNumNeurons(); j++) {
				nodeVal += layerB.m_vpNeurons[i]->getWeight(j) * vResultErrorRate[j];
				DeltaWjk.at<float>(i, j) = learningRate * vResultErrorRate[j] * layerB.m_vpNeurons[i]->getValue();
			}
			float sigmoid = sigmoidFunction(layerB.m_vpNeurons[i]->getValue());
			DeltaJ[i] = nodeVal * sigmoid * (1 - sigmoid);
		}

		for (size_t i = 0; i < layerA.getNumNeurons(); i++) {
			for (size_t j = 0; j < layerB.getNumNeurons(); j++) {
				float Delta = learningRate * DeltaJ[j] * layerA.m_vpNeurons[i]->getValue();
				float oldWeight = layerA.m_vpNeurons[i]->getWeight(j);
				layerA.m_vpNeurons[i]->setWeight(j, oldWeight + Delta);
			}
		}

		for (size_t i = 0; i < layerB.getNumNeurons(); i++) {
			for (size_t j = 0; j < layerC.getNumNeurons(); j++) {
				float oldWeight = layerB.m_vpNeurons[i]->getWeight(j);
				layerB.m_vpNeurons[i]->setWeight(j, oldWeight + DeltaWjk.at<float>(i, j));
			}
		}
	}


}}