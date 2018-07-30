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

#include "GraphDense.h"
#include "edgePotentialPotts.h"
#include "macroses.h"

// Constructor
CGraphDense::CGraphDense(byte nStates) : m_nStates(nStates)
{ }

// Destructor
CGraphDense::~CGraphDense(void)
{
	for (auto edgePot : m_vpEdgePots)
		delete edgePot;
}

void CGraphDense::setNodes(const float *pots, size_t nNodes)
{
	m_nNodes = nNodes;
    m_vUnary = vec_float_t(pots, pots + m_nNodes * m_nStates);
}

void CGraphDense::setEdgesPotts(const float *features, word nFeatures, float w, const SemiMetricFunction *function)
{
	if (function)	setEdges(new CEdgePotentialPottsSemiMetric(features, nFeatures, m_nNodes, w, function));
	else			setEdges(new CEdgePotentialPotts(features, nFeatures, m_nNodes, w));
}

void CGraphDense::setEdges(CEdgePotential *pEdgePot)
{
	m_vpEdgePots.push_back(pEdgePot);
}
