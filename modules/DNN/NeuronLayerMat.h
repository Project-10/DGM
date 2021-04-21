#pragma once

#include "Neuron.h"

namespace DirectGraphicalModels {
	namespace dnn
	{
		class CNeuronLayerMat
		{
		public:
			/**
			 * @brief Constructor
			 * @param numNeurons The number of neurons in the layer
			 * @param numConnections The number of incoming connections for every neuron
			 * @note In feed-forward networks \b numConnections is usually equal to the number of neurons on the previouse layer
			 */
			DllExport CNeuronLayerMat(size_t numNeurons, size_t numConnections)
				: m_values(numNeurons, 1, CV_32FC1)
				, m_weights(numConnections, numNeurons, CV_32FC1)
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


		private:
			Mat	m_values;	///< The values of the neurons at the layer (1d matrix)
			Mat m_weights;	///< The weight of the neurons (2d matrix )
		};
	}
}

