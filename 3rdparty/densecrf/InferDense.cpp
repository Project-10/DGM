#include "InferDense.h"
#include "../modules/DGM/GraphDense.h"
#include "fastmath.h"
#include "edgePotentialPotts.h"
#include <numeric>

namespace DirectGraphicalModels 
{
	void CInferDense::infer(unsigned int nIt)
	{
		float relax = 1.0f;

		startInference();

		for (unsigned int i = 0; i < nIt; i++)
			stepInference(relax);

		m_current.copyTo(getGraphDense()->m_nodePotentials);
	}

	namespace {
		void expAndNormalize(Mat &out, const Mat &in, float scale = 1.0f, float relax = 1.0f)
		{
			vec_float_t V(in.cols);
			for (int n = 0; n < in.rows; n++) {             // node
				const float *pIn = in.ptr<float>(n);
				float       *pOut = out.ptr<float>(n);

				// Find the max and subtract it so that the exp doesn't explode
				//  float mx = *std::max_element(in.begin() + i * m_pGraph->m_nStates, in.begin() + (i + 1) * m_pGraph->m_nStates);
				float max = scale * pIn[0];
				for (int s = 1; s < in.cols; s++)
					if (scale * pIn[s] > max) max = scale * pIn[s];

				for (size_t j = 0; j < V.size(); j++)
					V[j] = fast_exp(scale * pIn[j] - max);

				float sum = 0;
				for (float &v : V) sum += v;
				//      float sum = std::accumulate(V.begin(), V.end(), 0);

				// Make it a probability
				for (float &v : V) v /= sum;

				for (size_t j = 0; j < V.size(); j++)
					pOut[j] = (relax == 1) ? V[j] : (1 - relax) * pOut[j] + relax * V[j];
			}
		}
	}

	void CInferDense::startInference(void)
	{
		const int rows = getGraphDense()->m_nodePotentials.rows;
		const int cols = getGraphDense()->m_nodePotentials.cols;

		m_additionalUnary = Mat(rows, cols, CV_32FC1, Scalar(0));
		m_current = Mat(rows, cols, CV_32FC1, Scalar(0));
		m_next = Mat(rows, cols, CV_32FC1, Scalar(0));
		m_temp = Mat(2 * rows, cols, CV_32FC1, Scalar(0));

		// Making log potentials
		for (int n = 0; n < rows; n++) {
			float *pUnary = getGraphDense()->m_nodePotentials.ptr<float>(n);
			for (int s = 0; s < cols; s++)
				pUnary[s] = -logf(pUnary[s]);
		}

		// TODO: exp is not needed actually
		expAndNormalize(m_current, getGraphDense()->m_nodePotentials, -1.0f);            // Initialize using the unary energies
	}

	void CInferDense::stepInference(float relax)
	{
		// Set the unary potential
		m_next = -getGraphDense()->m_nodePotentials - m_additionalUnary;

		// Add up all pairwise potentials
		for (auto &edgePot : getGraphDense()->m_vpEdgeModels)
			edgePot->apply(m_next, m_current, m_temp);

		// Exponentiate and normalize
		expAndNormalize(m_current, m_next, 1.0f, relax);
	}


	void CInferDense::currentMap(short *result)
	{
		// Find the map
		for (int n = 0; n < m_current.rows; n++) { // nodes
			const float *pCurrent = m_current.ptr<float>(n);
			// Find the max and subtract it so that the exp doesn't explode
			float mx = pCurrent[0];
			int imx = 0;
			for (int j = 1; j < getGraph()->getNumStates(); j++)
				if (mx < pCurrent[j]) {
					mx = pCurrent[j];
					imx = j;
				}
			result[n] = imx;
		}
	}
}