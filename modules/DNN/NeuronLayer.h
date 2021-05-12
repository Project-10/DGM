#pragma once

#include "Neuron.h"

namespace DirectGraphicalModels {
	namespace dnn
	{
		class CNeuronLayer
		{
		public:
			/**
			 * @brief Constructor
			 * @param numNeurons The number of neurons in the layer
			 * @param numConnections The number of incoming connections for every neuron
			 * @note In feed-forward networks \b numConnections is usually equal to the number of neurons on the previouse layer
			 */
			DllExport CNeuronLayer(int numNeurons, int numConnections, const std::function<float(float x)>& activationFunction)
				: m_netValues(numNeurons, 1, CV_32FC1)
				, m_weights(numConnections, numNeurons, CV_32FC1)
				, m_activationFunction(activationFunction)
			{}
			DllExport CNeuronLayer(const CNeuronLayer&) = delete;
			DllExport ~CNeuronLayer(void) = default;

			DllExport bool      operator=(const CNeuronLayer&) = delete;

			DllExport void      generateRandomWeights(void);
			DllExport void      dotProd(const Mat& values);
			DllExport Mat		getValues(void) const;

			// Accessors
			DllExport void	setNetValues(const Mat& values);
			DllExport Mat	getNetValues(void) const { return m_netValues; }
			DllExport Mat	getWeights(void) const { return m_weights; }
			DllExport int   getNumNeurons(void) const { return m_netValues.rows; }


		private:
			Mat								m_netValues;			///< The values of the neurons at the layer (1d matrix)
			Mat								m_weights;				///< The weight of the neurons (2d matrix )
			std::function<float(float y)>	m_activationFunction;	///< The activation function
		};
	}
}

