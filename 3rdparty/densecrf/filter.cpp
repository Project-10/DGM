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
#include "filter.h"
#include "permutohedral.h"

// Constructor
CFilter::CFilter(const float *source_features, int N_source, const float *target_features, int N_target, int feature_dim)
    : m_n1(N_source)
    , m_o1(0)
    , m_n2(N_target)
    , m_o2(N_source)
{
    m_pPermutohedral = new CPermutohedral();
    float * features = new float[(N_source + N_target) * feature_dim];
    memcpy(features, source_features, N_source * feature_dim * sizeof(float));
    memcpy(features + N_source * feature_dim, target_features, N_target * feature_dim * sizeof(float));
    m_pPermutohedral->init(features, feature_dim, N_source + N_target);
    delete [] features;
}

// Constructor
CFilter::CFilter(const float *features, int N, int feature_dim )
    : m_n1(N)
    , m_o1(0)
    , m_n2(N)
    , m_o2(0)
{
    m_pPermutohedral = new CPermutohedral();
    m_pPermutohedral->init(features, feature_dim, N);
}

// Destructor
CFilter::~CFilter(void)
{
    delete m_pPermutohedral;
}

void CFilter::filter(const float *source, float *target, int value_size)
{
    m_pPermutohedral->compute(target, source, value_size, m_o1, m_o2, m_n1, m_n2);
}
