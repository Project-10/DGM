// This code was deeply revised by Sergey Kosov in 2018 - 2019 for Project X
// to support OpenCV and moderm C++11 standard
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

/************************************************/
/***          Permutohedral Lattice           ***/
/************************************************/
class CPermutohedral
{
public:
    CPermutohedral(void) = default;
    CPermutohedral(const CPermutohedral& rhs);
    CPermutohedral& operator= (const CPermutohedral& rhs);
	~CPermutohedral(void) = default;

    void init(const Mat& features);
    void compute(const Mat& src, Mat& dst, int in_offset = 0, int out_offset = 0, size_t in_size = 0, size_t out_size = 0) const;

    
private:
    int	m_nFeatures         = 0;        // Number of elements
    int	m_M                 = 0;        // Size of sparse discretized space
    int m_featureSize       = 0;        // Dimension of features
    
    Mat m_offset			= Mat();
    Mat	m_barycentric       = Mat();
    Mat	m_blurNeighbor1		= Mat();
	Mat	m_blurNeighbor2		= Mat();
};
