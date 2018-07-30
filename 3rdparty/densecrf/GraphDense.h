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

class CGraphDense {
	friend class CInferDense;

public:
	DllExport CGraphDense(byte nStates);
	DllExport virtual ~CGraphDense(void);

	// Set the unary potential for all variables and labels (memory order is [x0l0 x0l1 x0l2 .. x1l0 x1l1 ...])
	DllExport virtual void setNodes(const float *pots, size_t nNodes);

	// Add  a pairwise potential defined over some feature space
	// The potential will have the form:    w*exp(-0.5*|f_i - f_j|^2)
	// The kernel shape should be captured by transforming the
	// features before passing them into this function
	DllExport void setEdgesPotts(const float *features, word nFeatures, float w = 1.0f, const SemiMetricFunction *function = NULL);
	
	// Add your own favorite pairwise potential (ownwership will be transfered to this class)
	DllExport void setEdges(CEdgePotential *pEdgePot);
	
	
	/**
	* @brief Returns the number of nodes in the graph
	* @returns number of nodes
	*/
	DllExport virtual size_t	getNumNodes(void) const { return m_nNodes; }


private:
	size_t		m_nNodes;					// number of pixels
	byte		m_nStates;
	
    vec_float_t						m_vUnary;
	std::vector<CEdgePotential *>	m_vpEdgePots;


private:
	// Copy semantics are disabled
	CGraphDense(const CGraphDense &rhs) {}
	const CGraphDense & operator= (const CGraphDense & rhs) { return *this; }
};



