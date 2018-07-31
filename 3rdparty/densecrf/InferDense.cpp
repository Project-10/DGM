#include "InferDense.h"
#include "GraphDense.h"
#include "fastmath.h"
#include "edgePotentialPotts.h"

// TODO: getPotentials ?
vec_float_t CInferDense::infer(unsigned int nIt, float relax)
{
	vec_float_t res(m_pGraph->getNumNodes() * m_pGraph->m_nStates);
	
	// Run inference
	vec_float_t vProb = runInference(nIt, relax);

	// Copy the result over     // TODO:
	for (int i = 0; i < m_pGraph->getNumNodes(); i++)
		memcpy(res.data() + i * m_pGraph->m_nStates, vProb.data() + i * m_pGraph->m_nStates, m_pGraph->m_nStates * sizeof(float));

	return res;
}

// TODO: probably switch to the DGM decode 
vec_byte_t CInferDense::decode(unsigned int nIt, float relax)
{
	vec_byte_t res;
	res.reserve(m_pGraph->getNumNodes());

	// Run inference
	vec_float_t vProb = runInference(nIt, relax);

	// Find the map
	for (int i = 0; i < m_pGraph->getNumNodes(); i++) {
		const float *p = vProb.data() + i * m_pGraph->m_nStates;
		// Find the max and subtract it so that the exp doesn't explode
		float mx = p[0];
		byte imx = 0;
		for (byte j = 1; j < m_pGraph->m_nStates; j++)
			if (mx < p[j]) {
				mx = p[j];
				imx = j;
			}
		res.push_back(imx);
	}

	return res;
}

// TODO: infer() ?
vec_float_t CInferDense::runInference(unsigned int nIt, float relax)
{
	m_vAdditionalUnary.resize(m_pGraph->m_vUnary.size());
	std::fill(m_vAdditionalUnary.begin(), m_vAdditionalUnary.end(), 0);

	m_vCurrent.resize(m_pGraph->m_vUnary.size());
	std::fill(m_vCurrent.begin(), m_vCurrent.end(), 0);

	m_vNext.resize(m_pGraph->m_vUnary.size());
	std::fill(m_vNext.begin(), m_vNext.end(), 0);

	m_vTmp.resize(2 * m_pGraph->m_vUnary.size());
	std::fill(m_vTmp.begin(), m_vTmp.end(), 0);
	
	startInference();
	
	for (unsigned int i = 0; i < nIt; i++)
		stepInference(relax);
	
	return m_vCurrent;
}

void CInferDense::expAndNormalize(vec_float_t &out, const vec_float_t &in, float scale, float relax)
{
	float *V = new float[m_pGraph->getNumNodes() + 10];
	for (int i = 0; i<m_pGraph->getNumNodes(); i++) {
		const float * b = in.data() + i * m_pGraph->m_nStates;
		// Find the max and subtract it so that the exp doesn't explode
		float mx = scale * b[0];
		for (int j = 1; j<m_pGraph->m_nStates; j++)
			if (mx < scale*b[j])
				mx = scale * b[j];
		float tt = 0;
		for (int j = 0; j<m_pGraph->m_nStates; j++) {
			V[j] = fast_exp(scale*b[j] - mx);
			tt += V[j];
		}
		// Make it a probability
		for (int j = 0; j<m_pGraph->m_nStates; j++)
			V[j] /= tt;

		float *a = out.data() + i * m_pGraph->m_nStates;
		for (int j = 0; j < m_pGraph->m_nStates; j++)
			if (relax == 1)
				a[j] = V[j];
			else
				a[j] = (1 - relax)*a[j] + relax * V[j];
	}
	delete[] V;
}

void CInferDense::startInference(void)
{
	expAndNormalize(m_vCurrent, m_pGraph->m_vUnary, -1);			// Initialize using the unary energies
}

void CInferDense::stepInference(float relax)
{
	// Set the unary potential
	for (size_t i = 0; i < m_vNext.size(); i++)
		m_vNext[i] = -m_pGraph->m_vUnary[i] - m_vAdditionalUnary[i];

	// Add up all pairwise potentials
	for (auto &edgePot : m_pGraph->m_vpEdgePots)
		edgePot->apply(m_vNext, m_vCurrent, m_vTmp, m_pGraph->m_nStates);

	// Exponentiate and normalize
	expAndNormalize(m_vCurrent, m_vNext, 1.0f, relax);
}

void CInferDense::currentMap(short *result)
{
	// Find the map
	for (int i = 0; i < m_pGraph->getNumNodes(); i++) {
		const float * p = m_vCurrent.data() + i * m_pGraph->m_nStates;
		// Find the max and subtract it so that the exp doesn't explode
		float mx = p[0];
		int imx = 0;
		for (int j = 1; j<m_pGraph->m_nStates; j++)
			if (mx < p[j]) {
				mx = p[j];
				imx = j;
			}
		result[i] = imx;
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
