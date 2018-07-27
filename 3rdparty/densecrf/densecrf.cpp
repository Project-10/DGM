/*
    Copyright (c) 2011, Philipp Krähenbühl
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
        * Neither the name of the Stanford University nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY Philipp Krähenbühl ''AS IS'' AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL Philipp Krähenbühl BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "densecrf.h"
#include "fastmath.h"
#include "permutohedral.h"
#include "macroses.h"

// Constructor
DenseCRF::DenseCRF(byte nStates) : m_nStates(nStates)
{ }

DenseCRF::~DenseCRF(void)
{
	for(unsigned int i = 0; i < pairwise_.size(); i++ )
		delete pairwise_[i];
}

//////////////////////////////
/////  Unary Potentials  /////
//////////////////////////////
void DenseCRF::setNodes(const float *pots, int nNodes)
{
	m_nNodes = nNodes;
		
    m_vUnary = vec_float_t(pots, pots + m_nNodes * m_nStates);

    m_vAdditionalUnary.resize(m_nNodes * m_nStates);
    std::fill(m_vAdditionalUnary.begin(), m_vAdditionalUnary.end(), 0);
    
    m_vCurrent.resize(m_nNodes * m_nStates);
    std::fill(m_vCurrent.begin(), m_vCurrent.end(), 0);
    
	m_vNext.resize(m_nNodes * m_nStates);
	std::fill(m_vNext.begin(), m_vNext.end(), 0);

	m_vTmp.resize(2 * m_nNodes * m_nStates);
	std::fill(m_vTmp.begin(), m_vTmp.end(), 0);
}


/////////////////////////////////
/////  Pairwise Potentials  /////
/////////////////////////////////
void DenseCRF::addPairwiseEnergy (const float* features, int D, float w, const SemiMetricFunction * function) 
{
	if (function)	addPairwiseEnergy(new SemiMetricPotential( features, D, m_nNodes, w, function));
	else			addPairwiseEnergy(new PottsPotential(features, D, m_nNodes, w));
}

void DenseCRF::addPairwiseEnergy ( PairwisePotential* potential )
{
	pairwise_.push_back( potential );
}

///////////////////////
/////  Inference  /////
///////////////////////
void DenseCRF::infer(unsigned int nIt, float *result, float relax) 
{
	// Run inference
	vec_float_t vProb = runInference(nIt, relax);

	// Copy the result over
	for(int i = 0; i < m_nNodes; i++)
		memcpy(result + i * m_nStates, vProb.data() + i * m_nStates, m_nStates * sizeof(float));
}

vec_byte_t DenseCRF::decode(unsigned int nIt, float relax)
{
	vec_byte_t res;
	res.reserve(m_nNodes);

	// Run inference
	vec_float_t vProb = runInference(nIt, relax);
	
	// Find the map
	for(int i = 0; i < m_nNodes; i++) {
		const float *p = vProb.data() + i * m_nStates;
		// Find the max and subtract it so that the exp doesn't explode
		float mx = p[0];
		byte imx = 0;
		for(byte j = 1; j < m_nStates; j++)
			if(mx < p[j]) {
				mx = p[j];
				imx = j;
			}
		res.push_back(imx);
	}

	return res;
}

vec_float_t DenseCRF::runInference(unsigned int nIt, float relax)
{
	startInference();
	for(unsigned int i = 0; i < nIt; i++)
		stepInference(relax);
	return m_vCurrent;
}

void DenseCRF::expAndNormalize (vec_float_t &out, const vec_float_t &in, float scale, float relax)
{
	float *V = new float[ m_nNodes+10 ];
	for( int i=0; i<m_nNodes; i++ ){
		const float * b = in.data() + i*m_nStates;
		// Find the max and subtract it so that the exp doesn't explode
		float mx = scale*b[0];
		for( int j=1; j<m_nStates; j++ )
			if( mx < scale*b[j] )
				mx = scale*b[j];
		float tt = 0;
		for( int j=0; j<m_nStates; j++ ){
			V[j] = fast_exp( scale*b[j]-mx );
			tt += V[j];
		}
		// Make it a probability
		for( int j=0; j<m_nStates; j++ )
			V[j] /= tt;
		
		float *a = out.data() + i * m_nStates;
		for(int j = 0; j < m_nStates; j++)
			if (relax == 1)
				a[j] = V[j];
			else
				a[j] = (1-relax)*a[j] + relax*V[j];
	}
	delete[] V;
}
///////////////////
/////  Debug  /////
///////////////////
#ifdef DEBUG_MODE1
void DenseCRF::unaryEnergy(const short* ass, float* result)
{
	for(int i = 0; i < m_nNodes; i++ )
		if ( 0 <= ass[i] && ass[i] < m_nStates )
			result[i] = unary_[ m_nStates*i + ass[i] ];
		else
			result[i] = 0;
}

void DenseCRF::pairwiseEnergy(const short* ass, float* result, int term) 
{
	vec_float_t current(m_nNodes * m_nStates, 0);
	// Build the current belief [binary assignment]
	for( int i=0,k=0; i<m_nNodes; i++ )
		for( int j=0; j<m_nStates; j++, k++ )
			current[k] = (ass[i] == j);
	
	std::fill(m_vNext.begin(), m_vNext.end(), 0);

	if (term == -1)
		for( unsigned int i=0; i<pairwise_.size(); i++ )
			pairwise_[i]->apply(m_vNext, current, m_vTmp, m_nStates );
	else
		pairwise_[ term ]->apply(m_vNext, current, m_vTmp, m_nStates );
	for(int i = 0; i < m_nNodes; i++)
		if ( 0 <= ass[i] && ass[i] < m_nStates)	result[i] = -m_vNext[i * m_nStates + ass[i]];
		else									result[i] = 0;
}
#endif

void DenseCRF::startInference(void)
{
	expAndNormalize(m_vCurrent, m_vUnary, -1);			// Initialize using the unary energies
}

void DenseCRF::stepInference(float relax)
{
	// Set the unary potential
	for(size_t i = 0; i < m_vNext.size(); i++)
		m_vNext[i] = -m_vUnary[i] - m_vAdditionalUnary[i];
	
	// Add up all pairwise potentials
	for( unsigned int i=0; i<pairwise_.size(); i++ )
		pairwise_[i]->apply(m_vNext, m_vCurrent, m_vTmp, m_nStates);
	
	// Exponentiate and normalize
	expAndNormalize(m_vCurrent, m_vNext, 1.0f, relax);
}

void DenseCRF::currentMap(short *result)
{
	// Find the map
	for( int i=0; i<m_nNodes; i++ ){
		const float * p = m_vCurrent.data() + i * m_nStates;
		// Find the max and subtract it so that the exp doesn't explode
		float mx = p[0];
		int imx = 0;
		for( int j=1; j<m_nStates; j++ )
			if( mx < p[j] ){
				mx = p[j];
				imx = j;
			}
		result[i] = imx;
	}
}
