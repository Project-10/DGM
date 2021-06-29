#include "Perceptron.h"
#include "macroses.h"

namespace DirectGraphicalModels {
	namespace dnn {
		// Constructor
		CPerceptron::CPerceptron(const std::vector<int>& vNumNeurons) {
			// TODO: imp,lement this constructor in the future
			for (size_t i = 0; i < vNumNeurons.size(); i++){
				int numNeurons = vNumNeurons[i];
				int numConnections = (i == 0) ? 0 : vNumNeurons[i - 1];
				ptr_nl_t pNeuronLayer = std::make_shared<CNeuronLayer>(numNeurons, numConnections, [](float x) {return x; }, [](float x) { return 1; });
				m_vpNeuronLayers.push_back(pNeuronLayer);
			}
		}

		// Constructor
		CPerceptron::CPerceptron(const std::vector<ptr_nl_t>& vpLayers, const std::vector<ptr_nl_t>& numHiddenLayer)
		{
			//for (auto& nl : vpLayers)
			//	m_vpNeuronLayers.push_back(nl);

			m_vpNeuronLayers.push_back(vpLayers[0]);
			
			for (int i = 0; i < numHiddenLayer.size(); i++) 
				m_vpNeuronLayers.push_back(numHiddenLayer[i]);

			m_vpNeuronLayers.push_back(vpLayers[1]);
		}

		Mat	CPerceptron::getPrediction(const Mat& inputValues)
		{
			// Asserions
			DGM_ASSERT(m_vpNeuronLayers.size() > 1);

			// Initialize the values of the input layer
			m_vpNeuronLayers[0]->setNetValues(inputValues); // input layer

			// Calculate values of the nodes on all other layers
			for (size_t i = 1; i < m_vpNeuronLayers.size(); i++) {
				m_vpNeuronLayers[i]->dotProd(m_vpNeuronLayers[i - 1]->getValues()); // determining values of hidden layers neurons
			}

			// Return the node values from the output layer
			return m_vpNeuronLayers.back()->getValues(); 
		}

		// TODO: this method works for multiple layers, solution = guesses, gt = answers
		void CPerceptron::backPropagate(const Mat& solution, const Mat& gt, float learningRate, int numHiddenLayer)
		{
			int numLayers = m_vpNeuronLayers.size() - 1; // number of layers [0, 1, 2 ... n]
			
			Mat error = gt - solution;
			for (int i = 0; i < error.rows; i++) {
				error.at<float>(i, 0) *= m_vpNeuronLayers[numLayers]->getActivationFunctionDeriateve()(solution.at<float>(i, 0));
			}
			
			std::vector<Mat> Error; //Vector containing errors for each of the layers neurons
			Error.push_back(error);
			
			for (int i = 0; i < numHiddenLayer; i++)
			{
				Mat temp = m_vpNeuronLayers[numLayers-i]->getWeights() * Error[i]; // weights * error
				Error.push_back(temp);
			}

			for(int i = 1; i < m_vpNeuronLayers.size(); i++)
			{
				gemm(m_vpNeuronLayers[numLayers - i]->getValues(), Error[i-1].t(), learningRate, m_vpNeuronLayers[m_vpNeuronLayers.size() - i]->getWeights(), 1, m_vpNeuronLayers[m_vpNeuronLayers.size() - i]->getWeights());
			}
		}
	}
}