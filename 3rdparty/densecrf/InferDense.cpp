#include "InferDense.h"
#include "GraphDense.h"
#include "fastmath.h"
#include "edgePotentialPotts.h"
#include <numeric>

// TODO: probably switch to the DGM decode 
vec_byte_t CInferDense::decode(unsigned int nIt, float relax)
{
    vec_byte_t res;
	res.reserve(m_pGraph->getNumNodes());

	// Run inference
    vec_float_t vProb = infer(nIt, relax);
    
	// Find the map
	for (size_t i = 0; i < m_pGraph->getNumNodes(); i++) {
        auto it = std::max_element(vProb.begin() + i * m_pGraph->m_nStates, vProb.begin() +  (i + 1) * m_pGraph->m_nStates);
        byte idx = static_cast<byte>(std::distance(vProb.begin() + i * m_pGraph->m_nStates, it));   // index of the maximum element
		res.push_back(idx);
	}

	return res;
}

vec_float_t CInferDense::infer(unsigned int nIt, float relax)
{
    startInference();
	
	for (unsigned int i = 0; i < nIt; i++)
		stepInference(relax);
	
    float *pData = reinterpret_cast<float *>(m_current.data);
    return vec_float_t(pData, pData + m_current.rows * m_current.cols);
}

void CInferDense::startInference(void)
{
	const int rows = m_pGraph->m_unary.rows;
	const int cols = m_pGraph->m_unary.cols;
	
    m_additionalUnary   = Mat(rows, cols, CV_32FC1, Scalar(0));
    m_current           = Mat(rows, cols, CV_32FC1, Scalar(0));
    m_next              = Mat(rows, cols, CV_32FC1, Scalar(0));
    m_temp              = Mat(2 * rows, cols, CV_32FC1, Scalar(0));
    
	// Making log potentials
	for (int n = 0; n < rows; n++) {
		float *pUnary = m_pGraph->m_unary.ptr<float>(n);
		for (int s = 0; s < cols; s++)
			pUnary[s] = -logf(pUnary[s]);
	}
	
	// TODO: exp is not needed actually
	expAndNormalize(m_current, m_pGraph->m_unary, -1.0f);            // Initialize using the unary energies
}

void CInferDense::stepInference(float relax)
{
	// Set the unary potential
    // TODO: optimize
    for (int n = 0; n < m_next.rows; n++)
        for (int s = 0; s < m_next.cols; s++)
            m_next.at<float>(n, s) = -m_pGraph->m_unary.at<float>(n, s) - m_additionalUnary.at<float>(n, s);
    
    // Add up all pairwise potentials
    for (auto &edgePot : m_pGraph->m_vpEdgePots)
        edgePot->apply(m_next, m_current, m_temp, m_pGraph->m_nStates);
    
    // Exponentiate and normalize
    expAndNormalize(m_current, m_next, 1.0f, relax);
}

void CInferDense::expAndNormalize(Mat &out, const Mat &in, float scale, float relax) const
{
    vec_float_t V(in.cols);
    for (int n = 0; n < in.rows; n++) {             // node
        const float *pIn    = in.ptr<float>(n);
        float       *pOut   = out.ptr<float>(n);
		
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

void CInferDense::currentMap(short *result)
{
	// Find the map
	for (int n = 0; n < m_current.rows; n++) { // nodes
        const float *pCurrent = m_current.ptr<float>(n);
		// Find the max and subtract it so that the exp doesn't explode
		float mx = pCurrent[0];
		int imx = 0;
		for (int j = 1; j<m_pGraph->m_nStates; j++)
			if (mx < pCurrent[j]) {
				mx = pCurrent[j];
				imx = j;
			}
		result[n] = imx;
	}
}

///////////////////
/////  Debug  /////
///////////////////
#ifdef DEBUG_MODE1
void CGraphDense::unaryEnergy(const short* ass, float* result)
{
	for (int i = 0; i < m_nNodes; i++)
		if (0 <= ass[i] && ass[i] < m_nStates)
			result[i] = unary_[m_nStates*i + ass[i]];
		else
			result[i] = 0;
}

void CGraphDense::pairwiseEnergy(const short* ass, float* result, int term)
{
	vec_float_t current(m_nNodes * m_nStates, 0);
	// Build the current belief [binary assignment]
	for (int i = 0, k = 0; i<m_nNodes; i++)
		for (int j = 0; j<m_nStates; j++, k++)
			current[k] = (ass[i] == j);

	std::fill(m_vNext.begin(), m_vNext.end(), 0);

	if (term == -1)
		for (unsigned int i = 0; i<pairwise_.size(); i++)
			pairwise_[i]->apply(m_vNext, current, m_vTmp, m_nStates);
	else
		pairwise_[term]->apply(m_vNext, current, m_vTmp, m_nStates);
	for (int i = 0; i < m_nNodes; i++)
		if (0 <= ass[i] && ass[i] < m_nStates)	result[i] = -m_vNext[i * m_nStates + ass[i]];
		else									result[i] = 0;
}
#endif
