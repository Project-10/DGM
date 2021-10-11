#include "RBM.h"
#include "macroses.h"

namespace DirectGraphicalModels {
	namespace dnn {
		CRBM::CRBM(const std::vector<ptr_nl_t>& vpLayers){
			for (auto& nl : vpLayers)
				m_vpNeuronLayers.push_back(nl); //adding layers to the vector
		}

		void CRBM::setVisibleBias(float value) {
			visibleBias = value;                    //setting the bias for visible layer
		}

		void CRBM::setHiddenBias(float value) {
			hiddenBias = value;                     //setting the bias for hidden layer
		}

		Mat CRBM::feedForward(const Mat& values) { 
			/*
			In this function, we sample the hidden layer given the visible layer. 
			We apply the sigmoid function to sum of visible layer multiplied by weights.
			For now adding bias is not yet implemented.
			*/
			m_vpNeuronLayers[0]->setNetValues(values); //set the visible layer values

			m_vpNeuronLayers[1]->dotProd(m_vpNeuronLayers[0]->getValues()); //sigmoid(sum(visible * weights))  bias not implemented yet

			return m_vpNeuronLayers.back()->getValues();
		}

		Mat CRBM::feedBackward(void) const {
			/*
			Here we sample the visible layer given the hidden layer.
			We apply the sigmoid function to sum of hidden layer multiplied by weights.
			Adding bias is not yet implemented.
			*/
			
			m_vpNeuronLayers[0]->dotProd(m_vpNeuronLayers[1]->getValues()); //sigmoid(sum(hidden * weights))  bias not implemented yet
			//as we only have weights assigned to hidden layer, when we perform function above, it will result in a message saying
			//no weights initialized for visible layer, as a temporary solution I will add a function in Neuron layer class so it copies
			// hidden layers weight to visible layer, so they both have the same weights

			return m_vpNeuronLayers[0]->getValues();
		}

		void CRBM::contrastiveDivergence(const Mat& values, float learningRate, int stepsK) {
			/*
			We will apply the CD-K algorithm to change the weights and bias.
		
			       
			*/
		}
	}
}
