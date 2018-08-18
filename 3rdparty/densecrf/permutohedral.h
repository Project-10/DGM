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

struct Neighbors
{
    int n1, n2;
    Neighbors(int n1 = 0, int n2 = 0) : n1(n1), n2(n2) {}
};

/************************************************/
/***          Permutohedral Lattice           ***/
/************************************************/
class CPermutohedral
{
public:
    CPermutohedral(void) = default;
    CPermutohedral(const CPermutohedral &rhs);
    CPermutohedral & operator= (const CPermutohedral &rhs);
    ~CPermutohedral(void);

    void init(const float *pFeature, word nFeatures, size_t N);
    void compute(Mat &out, const Mat &in, int in_offset, int out_offset, size_t in_size, size_t out_size) const;

    
private:
    int			m_N                 = 0;        // Number of elements
    int			m_M                 = 0;        // Size of sparse discretized space
    word        m_nFeatures         = 0;        // Dimension of features
    
    int       * m_pOffset           = NULL;
    float     * m_pBarycentric      = NULL;
    Neighbors * m_pBlurNeighbors    = NULL;
};
