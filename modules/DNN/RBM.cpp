#include "RBM.h"
#include "macroses.h"

namespace DirectGraphicalModels {
	namespace dnn {
		CRBM::CRBM(const std::vector<ptr_nl_t>& vpLayers){
			for (auto& nl : vpLayers)
				m_vpNeuronLayers.push_back(nl);
		}

		void CRBM::setVisibleBias(int value) {
			visibleBias = value;
		}

		void CRBM::setHiddenBias(int value) {
			hiddenBias = value;
		}

		Mat CRBM::feedForward(const Mat& values) {

		}

		Mat CRBM::feedBackward(const Mat& values) {

		}
	}
}