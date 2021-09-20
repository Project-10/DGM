#pragma once

#include "types.h"
#include "NeuronLayer.h"

namespace DirectGraphicalModels {
	namespace dnn {
		class CRBM {
			public:
				DllExport CRBM(const std::vector<ptr_nl_t>& vpLayers);
				DllExport CRBM(const CRBM&) = delete;
				DllExport ~CRBM(void) = default;

				DllExport bool operator=(const CRBM&) = delete;

				DllExport void setVisibleBias(int value);
				DllExport void setHiddenBias(int value);
				DllExport int getVisibleBias(void) const { return visibleBias; }
				DllExport int getHiddenBias(void) const { return hiddenBias; }

				DllExport Mat feedForward(const Mat& values);
				DllExport Mat feedBackward(const Mat& values);

			private:
				std::vector<ptr_nl_t> m_vpNeuronLayers;
				int                   visibleBias;
				int                   hiddenBias;
		};
	}
}