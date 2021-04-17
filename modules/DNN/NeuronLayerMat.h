#pragma once

#include "Neuron.h"

namespace DirectGraphicalModels {
	namespace dnn
	{
		class CNeuronLayerMat
		{
		public:
			DllExport CNeuronLayerMat(size_t numNeurons, size_t numConnections)
				: m_values(numNeurons, 1, CV_32FC1)
				, m_weights(numNeurons, numConnections, CV_32FC1)
			{}
			DllExport CNeuronLayerMat(const CNeuronLayerMat&) = delete;
			DllExport ~CNeuronLayerMat(void) = default;

			DllExport bool      operator=(const CNeuronLayerMat&) = delete;

			DllExport void      generateRandomWeights(void);
			DllExport void      dotProd(const CNeuronLayerMat& layer);
            

			// TODO: move this method to a proper place
			DllExport static void      backPropagate(CNeuronLayerMat& layerA, CNeuronLayerMat& layerB, CNeuronLayerMat& layerC, const Mat& resultErrorRate, float learningRate);


			// Accessors
			DllExport void	setValues(const Mat& values);
			DllExport Mat   getValues(void) const { return m_values; }
			DllExport int   getNumNeurons(void) const { return m_values.rows; }
            DllExport Mat   getWeights(void) const { return m_weights; }


		private:
			Mat	m_values;	///< The values of the neurons at the layer (1d matrix)
			Mat m_weights;	///< The weight of the neurons (2d matrix )
		};
	}
}

