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

#pragma once

#include "types.h"

class CEdgePotential;
class SemiMetricFunction;

class DenseCRF {
public:
	DllExport DenseCRF(byte nStates);
	DllExport virtual ~DenseCRF(void);

	// Set the unary potential for all variables and labels (memory order is [x0l0 x0l1 x0l2 .. x1l0 x1l1 ...])
	DllExport virtual void setNodes(const float *pots, int nNodes);

	// Add  a pairwise potential defined over some feature space
	// The potential will have the form:    w*exp(-0.5*|f_i - f_j|^2)
	// The kernel shape should be captured by transforming the
	// features before passing them into this function
	DllExport void setEdgesPotts(const float *features, word nFeatures, float w = 1.0f, const SemiMetricFunction *function = NULL);
	
	// Add your own favorite pairwise potential (ownwership will be transfered to this class)
	DllExport void setEdges(CEdgePotential *pEdgePot);
	
	// Run inference and return the probabilities
	DllExport void infer(unsigned int nIt = 1, float *result = NULL, float relax = 1.0f);
	
	// Run MAP inference and return the map for each pixel
	DllExport vec_byte_t decode(unsigned int nIt = 0, float relax = 1.0);
	
	// Step by step inference
	DllExport void startInference();
	DllExport void stepInference(float relax = 1.0);
	DllExport void currentMap(short *result);
	
public: /* Debugging functions */
	// Compute the unary energy of an assignment
	DllExport void unaryEnergy( const short * ass, float * result );
	
	// Compute the pairwise energy of an assignment (half of each pairwise potential is added to each of it's endpoints)
	DllExport void pairwiseEnergy( const short * ass, float * result, int term=-1 );


private:
	int			m_nNodes;		// number of pixels
	byte		m_nStates;
	
    vec_float_t m_vUnary;
    vec_float_t m_vAdditionalUnary;
    vec_float_t m_vCurrent;
    vec_float_t	m_vTmp;
	vec_float_t m_vNext;


	// Store all pairwise potentials
	std::vector<CEdgePotential *> m_vpEdgePots;

	// Run inference and return the pointer to the result
	vec_float_t runInference(unsigned int nIt, float relax);

	// Auxillary functions
	void expAndNormalize(vec_float_t &out, const vec_float_t &in, float scale = 1.0f, float relax = 1.0f);


private:
	// Copy semantics are disabled
	DenseCRF(const DenseCRF &rhs) {}
	const DenseCRF & operator= (const DenseCRF & rhs) { return *this; }
};



