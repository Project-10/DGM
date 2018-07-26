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

// Constructor
DenseCRF::DenseCRF(int nNodes, byte nStates) : m_nNodes(nNodes), m_nStates(nStates) {
	unary_				= new float[m_nNodes * m_nStates];	memset(unary_, 0, m_nNodes * m_nStates * sizeof(float));
	additional_unary_	= new float[m_nNodes * m_nStates];	memset(additional_unary_, 0, m_nNodes * m_nStates * sizeof(float));
	current_			= new float[m_nNodes * m_nStates];	memset(current_, 0, m_nNodes * m_nStates * sizeof(float));
	next_				= new float[m_nNodes * m_nStates];	memset(next_, 0, m_nNodes * m_nStates * sizeof(float));
	tmp_				= new float[2 * m_nNodes*m_nStates];	memset(tmp_, 0, 2 * m_nNodes * m_nStates * sizeof(float));
}

DenseCRF::~DenseCRF() {
	if (unary_)		delete[] unary_;
	if (additional_unary_) delete[] additional_unary_;
	if (current_)	delete[] current_;
	if (next_)		delete[] next_;
	if (tmp_)		delete[] tmp_;
	for(unsigned int i = 0; i < pairwise_.size(); i++ )
		delete pairwise_[i];
}

/////////////////////////////////
/////  Pairwise Potentials  /////
/////////////////////////////////
void DenseCRF::addPairwiseEnergy (const float* features, int D, float w, const SemiMetricFunction * function) {
	if (function)
		addPairwiseEnergy( new SemiMetricPotential( features, D, m_nNodes, w, function ) );
	else
		addPairwiseEnergy( new PottsPotential( features, D, m_nNodes, w ) );
}

void DenseCRF::addPairwiseEnergy ( PairwisePotential* potential ){
	pairwise_.push_back( potential );
}

void DenseCRF2D::addPairwiseGaussian ( float sx, float sy, float w, const SemiMetricFunction * function ) {
	float * feature = new float [m_nNodes*2];
	for( int j=0; j<m_height; j++ )
		for( int i=0; i<m_width; i++ ){
			feature[(j*m_width+i)*2+0] = i / sx;
			feature[(j*m_width+i)*2+1] = j / sy;
		}
	addPairwiseEnergy( feature, 2, w, function );
	delete [] feature;
}

void DenseCRF2D::addPairwiseBilateral ( float sx, float sy, float sr, float sg, float sb, const unsigned char* im, float w, const SemiMetricFunction * function ) {
	float * feature = new float [m_nNodes*5];
	for( int j=0; j<m_height; j++ )
		for( int i=0; i<m_width; i++ ){
			feature[(j*m_width+i)*5+0] = i / sx;
			feature[(j*m_width+i)*5+1] = j / sy;
			feature[(j*m_width+i)*5+2] = im[(i+j*m_width)*3+0] / sr;
			feature[(j*m_width+i)*5+3] = im[(i+j*m_width)*3+1] / sg;
			feature[(j*m_width+i)*5+4] = im[(i+j*m_width)*3+2] / sb;
		}
	addPairwiseEnergy( feature, 5, w, function );
	delete [] feature;
}
//////////////////////////////
/////  Unary Potentials  /////
//////////////////////////////
void DenseCRF::setNodes(const Mat &pots)
{
	memcpy( unary_, reinterpret_cast<const float *>(pots.data), m_nNodes*m_nStates*sizeof(float) );
}

///////////////////////
/////  Inference  /////
///////////////////////
void DenseCRF::infer(unsigned int nIt, float *result, float relax) 
{
	// Run inference
	float *prob = runInference(nIt, relax);
	// Copy the result over
	for(int i = 0; i < m_nNodes; i++)
		memcpy(result + i * m_nStates, prob + i * m_nStates, m_nStates * sizeof(float));
}

vec_byte_t DenseCRF::decode(unsigned int nIt, float relax)
{
	vec_byte_t res;
	res.reserve(m_nNodes);

	// Run inference
	float *prob = runInference(nIt, relax);
	
	// Find the map
	for(int i = 0; i < m_nNodes; i++) {
		const float * p = prob + i*m_nStates;
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

float* DenseCRF::runInference(unsigned int nIt, float relax )
{
	startInference();
	for(unsigned int i = 0; i < nIt; i++)
		stepInference(relax);
	return current_;
}

void DenseCRF::expAndNormalize ( float* out, const float* in, float scale, float relax ) {
	float *V = new float[ m_nNodes+10 ];
	for( int i=0; i<m_nNodes; i++ ){
		const float * b = in + i*m_nStates;
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
		
		float * a = out + i*m_nStates;
		for( int j=0; j<m_nStates; j++ )
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

void DenseCRF::unaryEnergy(const short* ass, float* result) {
	for( int i=0; i<m_nNodes; i++ )
		if ( 0 <= ass[i] && ass[i] < m_nStates )
			result[i] = unary_[ m_nStates*i + ass[i] ];
		else
			result[i] = 0;
}

void DenseCRF::pairwiseEnergy(const short* ass, float* result, int term) {
	float * current = new float[m_nNodes * m_nStates];
	memset(current, 0, m_nNodes * m_nStates * sizeof(float));
	// Build the current belief [binary assignment]
	for( int i=0,k=0; i<m_nNodes; i++ )
		for( int j=0; j<m_nStates; j++, k++ )
			current[k] = (ass[i] == j);
	
	for( int i=0; i<m_nNodes*m_nStates; i++ )
		next_[i] = 0;
	if (term == -1)
		for( unsigned int i=0; i<pairwise_.size(); i++ )
			pairwise_[i]->apply( next_, current, tmp_, m_nStates );
	else
		pairwise_[ term ]->apply( next_, current, tmp_, m_nStates );
	for( int i=0; i<m_nNodes; i++ )
		if ( 0 <= ass[i] && ass[i] < m_nStates )
			result[i] =-next_[ i*m_nStates + ass[i] ];
		else
			result[i] = 0;
	delete [] current;
	current = NULL;
}

void DenseCRF::startInference(){
	// Initialize using the unary energies
	expAndNormalize( current_, unary_, -1 );
}

void DenseCRF::stepInference( float relax ){
#ifdef SSE_DENSE_CRF
	__m128 * sse_next_ = (__m128*)next_;
	__m128 * sse_unary_ = (__m128*)unary_;
	__m128 * sse_additional_unary_ = (__m128*)additional_unary_;
#endif
	// Set the unary potential
#ifdef SSE_DENSE_CRF
	for( int i=0; i<(m_nNodes*m_nStates-1)/4+1; i++ )
		sse_next_[i] = - sse_unary_[i] - sse_additional_unary_[i];
#else
	for( int i=0; i<m_nNodes*m_nStates; i++ )
		next_[i] = -unary_[i] - additional_unary_[i];
#endif
	
	// Add up all pairwise potentials
	for( unsigned int i=0; i<pairwise_.size(); i++ )
		pairwise_[i]->apply( next_, current_, tmp_, m_nStates );
	
	// Exponentiate and normalize
	expAndNormalize( current_, next_, 1.0, relax );
}

void DenseCRF::currentMap( short * result ){
	// Find the map
	for( int i=0; i<m_nNodes; i++ ){
		const float * p = current_ + i*m_nStates;
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
