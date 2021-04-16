#pragma once

#include "Neuron.h"

namespace DirectGraphicalModels {
	namespace dnn
	{
		class CNeuronLayerMat
		{
		public:
			DllExport CNeuronLayerMat(size_t numNeurons, size_t numConnection);
			DllExport CNeuronLayerMat(const CNeuronLayerMat&) = delete;
			DllExport ~CNeuronLayerMat(void) = default;

			DllExport bool      operator=(const CNeuronLayerMat&) = delete;

			DllExport void      generateRandomWeights(void);
			DllExport void      setValues(const Mat& values);
			DllExport Mat       getValues(void) const;
			DllExport void      dotProd(const CNeuronLayerMat& layer);

			// TODO: move this method to a proper place
			DllExport static void      backPropagate(CNeuronLayerMat& layerA, CNeuronLayerMat& layerB, CNeuronLayerMat& layerC, std::vector<float>& vResultErrorRate, float learningRate);


			// Accessors
			DllExport size_t    getNumNeurons(void) const { return m_vpNeurons.size(); }


		private:
			std::vector<ptr_neuron_t>   m_vpNeurons;
		};
	}
}

