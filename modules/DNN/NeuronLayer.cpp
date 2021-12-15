#include "NeuronLayer.h"
#include "DGM/random.h"
#include "DGM/parallel.h"
#include "macroses.h"

namespace DirectGraphicalModels {
	namespace dnn
	{
		void CNeuronLayer::generateRandomWeights(void)
		{
			m_weights = random::U(m_weights.size(), m_weights.type(), -0.5f, 0.5f);
			m_biases = random::U(m_biases.size(), m_biases.type(), -0.5f, 0.5f);
		}

		void CNeuronLayer::dotProd(const Mat& values)
		{
			// this->m_netValues = this->m_weights * values + m_biases;
			gemm(m_weights.t(), values, 1, m_biases, 1, m_netValues);
		}
		void CNeuronLayer::dotProdVis(const Mat& values, const Mat& weights)
		{
			// this->m_netValues = weights * values + m_biases;
			gemm(weights, values, 1, m_biases, 1, m_netValues);
		}
		void CNeuronLayer::setNetValues(const Mat& values)
		{
			// Assertions
			DGM_ASSERT(values.type() == m_netValues.type());
			DGM_ASSERT(values.size() == m_netValues.size());
			values.copyTo(m_netValues);
		}

		Mat CNeuronLayer::getValues(void) const
		{
			Mat res(m_netValues.clone());
			for (int y = 0; y < res.rows; y++) {
				float* pRes = res.ptr<float>(y);
				for (int x = 0; x < res.cols; x++)
					pRes[x] = m_activationFunction(pRes[x]);
			}
			return res;
		}

	}
}