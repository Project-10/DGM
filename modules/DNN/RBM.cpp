#include "RBM.h"
#include "macroses.h"

namespace DirectGraphicalModels {
	namespace dnn {
		CRBM::CRBM(const std::vector<ptr_nl_t>& vpLayers){
			for (auto& nl : vpLayers)
				m_vpNeuronLayers.push_back(nl);
		}

		void CRBM::setVisibleBias(float value) {
			visibleBias = value;
		}

		void CRBM::setHiddenBias(float value) {
			hiddenBias = value;
		}

		Mat CRBM::feedForward(const Mat& values) {
			m_vpNeuronLayers[0]->setNetValues(values); //set the visible layer values

			m_vpNeuronLayers[1]->dotProd(m_vpNeuronLayers[0]->getValues()); //sigmoid(sum(visible * weights))  bias not implemented yet

			return m_vpNeuronLayers.back()->getValues();
		}

		Mat CRBM::feedBackward(void) const {
			m_vpNeuronLayers[0]->dotProd(m_vpNeuronLayers[1]->getValues()); //sigmoid(sum(hidden * weights))  bias not implemented yet
			//as we only have weights assigned to hidden layer, when we perform function above, it will result in a message saying
			//no weights initialized for visible layer, as a temporary solution I will add a function in Neuron layer class so it copies
			// hidden layers weight to visible layer, so they both have the same weights

			return m_vpNeuronLayers[0]->getValues();
		}

		void CRBM::contrastiveDivergence(const Mat& values, float learningRate, int stepsK) {

		}
	}
}