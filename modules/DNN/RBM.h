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

				DllExport void debug(); //printing out rows and cols just to be sure they are correct

				DllExport Mat getBinomial(Mat mean);
				DllExport Mat propagateUp( Mat values); //hidden layer = sigmoid(sum(visible*weights)+bias)
				DllExport Mat propagateDown(Mat values); //visible layer = sigmoid(sum(hidden*weights)+bias)

				DllExport void sampleVisible(Mat values); //sample viible for negative phase
				DllExport void sampleHiddenPositive(Mat values); //sample hidden for positive phase
				DllExport void sampleHiddenNegative(Mat values); //sample hidden for negative phase

				DllExport void gibbsHVH(Mat hiddenSample); //gibbs sampling(sample visible and then sample hidden negative phase)
				DllExport void contrastiveDivergence(const Mat& values, float learningRate);
				DllExport Mat reconstruct(Mat values);
				//DllExport void gibbsHVH()

			private:
				std::vector<ptr_nl_t>    m_vpNeuronLayers;

				Mat						 m_positiveHMean;
				Mat						 m_positiveHSample;
		
				Mat						 m_negativeHMean;
				Mat						 m_negativeHSample;

				Mat						 m_negativeVMean;
				Mat						 m_negativeVSample;
		};
	}
}