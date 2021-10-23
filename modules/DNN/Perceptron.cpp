#include "Perceptron.h"
#include "macroses.h"

namespace DirectGraphicalModels {
	namespace dnn {
		// Constructor
		CPerceptron::CPerceptron(const std::vector<int>& vNumNeurons) {
			// TODO: implement this constructor in the future
			for (size_t i = 0; i < vNumNeurons.size(); i++) {
				int numNeurons = vNumNeurons[i];
				int numConnections = (i == 0) ? 0 : vNumNeurons[i - 1];
				ptr_nl_t pNeuronLayer = std::make_shared<CNeuronLayer>(numNeurons, numConnections, [](float x) { return x; }, [](float x) { return 1.0f; });
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
		// dCost/dw = dCost/dNode.Value * dNode.Value/dNode.NetValue * dNode.NetValue/dNode.Weight
		// dCost/dw = 2(solution - gt) * ActivationFunctionDeriateve(nodeNetValue) * Node_i-1.Value
		void CPerceptron::backPropagate(const Mat& gt, float learningRate)
		{
			const int nLayers = static_cast<int>(m_vpNeuronLayers.size());
			
			// Assertion
			DGM_ASSERT_MSG(nLayers >= 3, "Percepton must contain at least 3 layers");
			
			vec_mat_t vDeltas(nLayers - 1);
			
			// compute init delta
			vDeltas.back() = m_vpNeuronLayers.back()->getValues() - gt;
			for (int i = 0; i < vDeltas.back().rows; i++)
				vDeltas.back().at<float>(i, 0) *= m_vpNeuronLayers.back()->getActivationFunctionDeriateve()(m_vpNeuronLayers.back()->getNetValues().at<float>(i, 0));
			
			// compute deltas
			for (int l = nLayers - 3; l >= 0; l--) {
				vDeltas[l] = m_vpNeuronLayers[l + 2]->getWeights() * vDeltas[l + 1];
				for (int i = 0; i < vDeltas[l].rows; i++)
					vDeltas[l].at<float>(i, 0) *= m_vpNeuronLayers[l + 1]->getActivationFunctionDeriateve()(m_vpNeuronLayers[l + 1]->getNetValues().at<float>(i, 0));
			}
		
			// compute gradient descen
			for (int l =  1; l < nLayers; l++)
				// Wi -= learningRate * x_(i-1) x delta_(i - 1).t();
				gemm(m_vpNeuronLayers[l - 1]->getValues(), vDeltas[l - 1].t(), -learningRate, m_vpNeuronLayers[l]->getWeights(), 1, m_vpNeuronLayers[l]->getWeights());
		}
	}
}
