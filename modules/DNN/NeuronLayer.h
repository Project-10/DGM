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
			DllExport CNeuronLayer(int numNeurons, int numConnections, const std::function<float(float x)>& activationFunction, const std::function<float(float x)>& activationFunctionDerivative)
				: m_netValues(numNeurons, 1, CV_32FC1)
				, m_weights(numConnections, numNeurons, CV_32FC1)
				, m_biases(numNeurons, 1, CV_32FC1)
				, m_activationFunction(activationFunction)
				, m_activationFunctionDerivative(activationFunctionDerivative)
			{}
			DllExport CNeuronLayer(const CNeuronLayer&) = delete;
			DllExport ~CNeuronLayer(void) = default;

			DllExport bool   operator=(const CNeuronLayer&) = delete;

			/**
			* @brief Fills the weight matrix and the biases vector with the random values in range (-0.5; 0.5)
			*/
			DllExport void	generateRandomWeights(void);
			/**
			* @brief Calulates the dot product between the neurons weights matrix and the input values vactor
			* @param values The input values vector from the previouse layer (size: 1 x numConnections; type: CV_32FC1)
			* @details This function updates the neurons' values as \f$ netValues_{(1\times N)} = weights^{\top}_{(C\times N)}\times values_{(1\times C)} + biases_{((1\times N))}\f$
			* @note This method updates only the nodes' net values
			*/
			DllExport void  dotProd(const Mat& values);
			/**
			* In the dotProd method, we cant multiply hidden neuron values by hidden neuron weights, so dotProdVis is created
			* which we can input by which weights neuron are multiplied by. 
			* in dotProd ->     this->m_netValues = this->m_weights * values + this->m_biases;
			* in dotProdVis ->  this->m_netValues = m_weights * values + this->m_biases;
			*/
			DllExport void  dotProdVis(const Mat& values, const Mat& weights); 
			/**
			* @brief Returns the values of the neurons of the layer
			* @note This method returns the result of per-element application of the activation function to the neurons' net values, i.e. activationFunction(netValues)
			* @returns The values of the neurons of the layer (size: 1 x numNeurons; type: CV_32FC1)
			*/
			DllExport Mat	getValues(void) const;

			// Accessors
			DllExport void	setNetValues(const Mat& values);
			DllExport Mat	getNetValues(void) const { return m_netValues; }
			DllExport Mat	getWeights(void) const { return m_weights; }
			DllExport Mat	getBiases(void) const { return m_biases; }
			DllExport int   getNumNeurons(void) const { return m_netValues.rows; }
			DllExport std::function<float(float y)> getActivationFunctionDeriateve(void) const { return m_activationFunctionDerivative; }


		private:
			Mat								m_netValues;					///< The values of the neurons at the layer (1d column-matrix) (size: 1 x numNeurons; type: CV_32FC1)
			Mat								m_weights;						///< The (incoming) weights of the neurons (2d matrix ) (size: numNeurons x numConnections; type: CV_32FC1)
			Mat								m_biases;						///< The biases of the neurons (1d column-matrix) (size: 1 x numNeurons; type: CV_32FC1)
			std::function<float(float y)>	m_activationFunction;			///< The activation function
			std::function<float(float y)>	m_activationFunctionDerivative;	///< The derivative of the activation function
		};

		using ptr_nl_t = std::shared_ptr<CNeuronLayer>;
	}
}
