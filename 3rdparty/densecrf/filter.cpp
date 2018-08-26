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
#include "macroses.h"

// Constructor
CFilter::CFilter(const Mat &src_features, const Mat &dst_features)
    : m_n1(src_features.rows)
    , m_o1(0)
    , m_n2(dst_features.rows)
    , m_o2(src_features.rows)
{
	// Assertions
	DGM_ASSERT(src_features.cols == dst_features.cols);
	
	m_pPermutohedral = new CPermutohedral();
	
	Mat features(src_features.rows + dst_features.rows, src_features.cols, CV_32FC1);
	src_features.copyTo(features(Rect(0, 0, src_features.cols, src_features.rows)));
	dst_features.copyTo(features(Rect(0, src_features.rows, dst_features.cols, dst_features.rows)));
	
    m_pPermutohedral->init(features);
}

// Constructor
CFilter::CFilter(const Mat &features)
    : m_n1(features.rows)
    , m_o1(0)
    , m_n2(features.rows)
    , m_o2(0)
{
    m_pPermutohedral = new CPermutohedral();
    m_pPermutohedral->init(features);
}

// Destructor
CFilter::~CFilter(void)
{
    delete m_pPermutohedral;
}

void CFilter::filter(const Mat &src, Mat &dst)
{
    m_pPermutohedral->compute(src, dst, m_o1, m_o2, m_n1, m_n2);
}
