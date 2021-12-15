#include "RBM.h"
#include "DGM/random.h"
#include "macroses.h"

namespace DirectGraphicalModels {
	namespace dnn {
		CRBM::CRBM(const std::vector<ptr_nl_t>& vpLayers){
			for (auto& nl : vpLayers)
				m_vpNeuronLayers.push_back(nl);
		}

		Mat CRBM::getBinomial(const Mat& mean) {
			Mat res(mean.clone());
			for (int y = 0; y < res.rows; y++) {
				float* pRes = res.ptr<float>(y);
				for (int x = 0; x < res.cols; x++) {
					if (pRes[x] < 0 || pRes[x]>1) {
						pRes[x] = 0;
					}
					double r = random::U<double>();	// uniformly distributed random number betwee 0 and 1
					if (r < pRes[x])
					{
						pRes[x] = 1;
					}
					else
					{
						pRes[x] = 0;
					}
				}
			}
			return res;
		}

		void CRBM::debug() {
			std::cout << "Weight - rows: " << m_vpNeuronLayers[1]->getWeights().rows << " cols: " << m_vpNeuronLayers[1]->getWeights().cols << std::endl;

			std::cout << "Positive H mean - rows: " << m_positiveHMean.rows << " cols: " << m_positiveHMean.cols << std::endl;
			std::cout << "Positive H sample - rows: " << m_positiveHSample.rows << " cols: " << m_positiveHSample.cols << std::endl;
			std::cout << "Negative H mean - rows: " << m_negativeHMean.rows << " cols: " << m_negativeHMean.cols << std::endl;
			std::cout << "Negative H sample - rows: " << m_negativeHSample.rows << " cols: " << m_negativeHSample.cols << std::endl;
			std::cout << "Negative V mean - rows: " << m_negativeVMean.rows << " cols: " << m_negativeVMean.cols << std::endl;
			std::cout << "Negative V sample - rows: " << m_negativeVSample.rows << " cols: " << m_negativeVSample.cols << std::endl;    
		}

		void CRBM::sampleVisible(const Mat& values) {
			m_negativeVMean = propagateDown(values);
			m_negativeVSample = getBinomial(m_negativeVMean);
		}

		void CRBM::sampleHiddenPositive(const Mat& values) {
			m_positiveHMean = propagateUp(values);		
			m_positiveHSample = getBinomial(m_positiveHMean);

			/*for (int y = 0; y < sample.rows; y++) {
				float* pRess = sample.ptr<float>(y);
				for (int x = 0; x < sample.cols; x++)
					std::cout << pRess[x] << std::endl;
			}*/
		}

		void CRBM::sampleHiddenNegative(const Mat& values) {
			m_negativeHMean = propagateUp(values);
			m_negativeHSample = getBinomial(m_negativeHMean);
		}

		Mat CRBM::propagateUp(const Mat& values) {
			m_vpNeuronLayers[0]->setNetValues(values); //set the visible layer values

			m_vpNeuronLayers[1]->dotProd(m_vpNeuronLayers[0]->getValues()); //sigmoid(sum(visible * weights)+bias)

			return m_vpNeuronLayers.back()->getValues();
		}

		Mat CRBM::propagateDown(const Mat& values){
			m_vpNeuronLayers[0]->dotProdVis(values, m_vpNeuronLayers[1]->getWeights()); 

			return m_vpNeuronLayers[0]->getValues();
		}

		void CRBM::gibbsHVH(const Mat& hiddenSample) {
			sampleVisible(hiddenSample);
			sampleHiddenNegative(m_negativeVSample);
		}
		/* This implementation of RBM uses single step contrastive divergence algorithm, called CD-1  */
		void CRBM::contrastiveDivergence(const Mat& values, float learningRate) {
			//-------POSITIVE PHASE--------------------
			/*In the positive phase, the input sample “v” from the visible layer is “clamped” to the input layer, 
			and then is propagated to the hidden layer. The result of the hidden layer activation is h.    */
			sampleHiddenPositive(values);

			//------NEGATIVE PHASE---------------------
			/*In the negative phase, “h” from the hidden layer is propagated back to the visible layer with the 
			new v, say v’. This is then propagated back to the hidden layer with activation result “h”    */
			gibbsHVH(m_positiveHMean);

			std::vector<double> test = m_negativeHSample;
			for (int i = 0; i < m_vpNeuronLayers[1]->getNumNeurons(); i++) {
				//std::cout << i << std::endl;
				for (int j = 0; j < m_vpNeuronLayers[0]->getNumNeurons(); j++)
				{
					m_vpNeuronLayers[1]->getWeights().at<float>(j, i) +=
						learningRate * (m_positiveHMean.at<float>(i, 0) * values.at<float>(j, 0) - m_negativeHMean.at<float>(i, 0) * m_negativeVSample.at<float>(j, 0))/4000; // divide
				}
				m_vpNeuronLayers[1]->getBiases().at<float>(i, 0) += learningRate * (m_positiveHSample.at<float>(i, 0) - m_negativeHMean.at<float>(i, 0))/4000; //divide
			}
			for (int i = 0; i < m_vpNeuronLayers[0]->getNumNeurons(); i++)
			{
				//std::cout << i << std::endl;
				m_vpNeuronLayers[0]->getBiases().at<float>(i, 0) += learningRate * (values.at<float>(i, 0) * m_negativeVSample.at<float>(i, 0))/4000; //divide
			}
		}

		Mat CRBM::reconstruct(const Mat& values) {
			Mat h, temp;

			h = propagateUp(values);
			temp = propagateDown(h);
			return temp;
		}
	}
}