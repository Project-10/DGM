#pragma once

#include "Neuron.h"

namespace DirectGraphicalModels {
	namespace dnn
	{
		class CNeuronLayerBias
		{
		public:
			/**
			 * @brief Constructor
			 * @param numNeurons The number of neurons in the layer
			 * @param numConnections The number of incoming connections for every neuron
			 * @note In feed-forward networks \b numConnections is usually equal to the number of neurons on the previouse layer
			 */
			DllExport CNeuronLayerBias(size_t numNeurons, size_t numConnections)
				: m_values(numNeurons, 1, CV_32FC1)
				, m_weights(numConnections + 1, numNeurons, CV_32FC1)
			{}
			DllExport CNeuronLayerBias(const CNeuronLayerBias&) = delete;
			DllExport ~CNeuronLayerBias(void) = default;

			DllExport bool      operator=(const CNeuronLayerBias&) = delete;

			DllExport void      generateRandomWeights(void);
			/**
			 * @note This method updates only the node values
			 */
			DllExport void      dotProd(const CNeuronLayerBias& layer);

			/**
			 * @note This method updates only weights of layerB and layerC
			 * @todo move this method to a proper place
			 */
			DllExport static void backPropagate(const CNeuronLayerBias& layerA, CNeuronLayerBias& layerB, CNeuronLayerBias& layerC, const Mat& resultErrorRate, float learningRate);


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


