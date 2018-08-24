#include "InferDense.h"

namespace DirectGraphicalModels
{
	namespace {
		void expAndNormalize(Mat &out, const Mat &in, float scale = 1.0f)
		{
			vec_float_t V(in.cols);
			for (int n = 0; n < in.rows; n++) {             // node
				const float *pIn  = in.ptr<float>(n);
				float       *pOut = out.ptr<float>(n);

				// Find the max and subtract it so that the exp doesn't explodeh
				float max = scale * pIn[0];
				for (int s = 1; s < in.cols; s++)
					if (scale * pIn[s] > max) max = scale * pIn[s];

				for (size_t j = 0; j < V.size(); j++)
					V[j] = expf(scale * pIn[j] - max);

				float sum = 0;
				for (float &v : V) sum += v;
				//      float sum = std::accumulate(V.begin(), V.end(), 0);

				// Make it a probability
				for (float &v : V) v /= sum;

				for (size_t j = 0; j < V.size(); j++)
					pOut[j] = V[j];
			}
		}
	}
	
	void CInferDense::infer(unsigned int nIt)
	{
		// ====================================== Initialization ======================================
		const int rows = getGraphDense()->m_nodePotentials.rows;
		const int cols = getGraphDense()->m_nodePotentials.cols;

		Mat current = Mat(rows, cols, CV_32FC1, Scalar(0));
		Mat next;
		Mat temp = Mat(2 * rows, cols, CV_32FC1, Scalar(0));

		// Making log potentials
		for (int n = 0; n < rows; n++) {
			float *pUnary = getGraphDense()->m_nodePotentials.ptr<float>(n);
			for (int s = 0; s < cols; s++)
				pUnary[s] = -logf(pUnary[s]);
		}

		// TODO: exp is not needed actually
		// Initialize using the unary energies
		expAndNormalize(current, getGraphDense()->m_nodePotentials, -1.0f);				// current = f(m_nodePotentials)  

		// =================================== Calculating potentials ==================================	
		for (unsigned int i = 0; i < nIt; i++) {
			// Set the unary potential
			next = -getGraphDense()->m_nodePotentials;

			// Add up all pairwise potentials
			for (auto &edgePotModel : getGraphDense()->m_vpEdgeModels)
				edgePotModel->apply(next, current, temp);								// next = f(next, current)

			// Exponentiate and normalize
			expAndNormalize(current, next, 1.0f);										// current = f(next)
		} // iter

		current.copyTo(getGraphDense()->m_nodePotentials);
	}
}