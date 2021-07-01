#include "Perceptron.h"
#include "macroses.h"

namespace DirectGraphicalModels {
	namespace dnn {
		// Constructor
		CPerceptron::CPerceptron(const std::vector<int>& vNumNeurons) {
			// TODO: imp,lement this constructor in the future
			for (size_t i = 0; i < vNumNeurons.size(); i++) {
				int numNeurons = vNumNeurons[i];
				int numConnections = (i == 0) ? 0 : vNumNeurons[i - 1];
				ptr_nl_t pNeuronLayer = std::make_shared<CNeuronLayer>(numNeurons, numConnections, [](float x) {return x; }, [](float x) { return 1; });
				m_vpNeuronLayers.push_back(pNeuronLayer);
			}
		}
		
		// Constructor
		CPerceptron::CPerceptron(const std::vector<ptr_nl_t>& vpLayers)
		{
			for (auto& nl : vpLayers)
				m_vpNeuronLayers.push_back(nl);
		}

		Mat	CPerceptron::getPrediction(const Mat& inputValues) 
		{
			// Asserions
			DGM_ASSERT(m_vpNeuronLayers.size() > 1);
			
			// Initialize the values of the input layer
			m_vpNeuronLayers[0]->setNetValues(inputValues);

			// Calculate values of the nodes on all other layers
			for (size_t i = 1; i < m_vpNeuronLayers.size(); i++)
				m_vpNeuronLayers[i]->dotProd(m_vpNeuronLayers[i - 1]->getValues());

			// Return the node values from the output layer
			return m_vpNeuronLayers.back()->getValues();
		}

		// TODO: this method works only for 3 layers
		void CPerceptron::backPropagate(const Mat& solution, const Mat& gt, float learningRate)
		{
			Mat error = gt - solution;
			for (int i = 0; i < error.rows; i++)
				error.at<float>(i, 0) *= m_vpNeuronLayers[2]->getActivationFunctionDeriateve()(solution.at<float>(i, 0));

			int numLayers = m_vpNeuronLayers.size() - 1; // number of layers [0, 1, 2 ... n]
			int numHiddenLayers = numLayers - 1;
			
			std::vector<Mat> Error; //Vector containing errors for each of the layers neurons
			Error.push_back(error);

			for (int i = 0; i < numHiddenLayers; i++)
			{
				Mat temp = m_vpNeuronLayers[numLayers - i]->getWeights() * Error[i]; // weights * error
				Error.push_back(temp);
			}

			for (int i = 1; i < m_vpNeuronLayers.size(); i++)
			{
				gemm(m_vpNeuronLayers[numLayers - i]->getValues(), Error[i - 1].t(), learningRate, m_vpNeuronLayers[m_vpNeuronLayers.size() - i]->getWeights(), 1, m_vpNeuronLayers[m_vpNeuronLayers.size() - i]->getWeights());
			}
		}
	}
}