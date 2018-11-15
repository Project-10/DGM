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
#include "permutohedral.h"
#include "hashtable.h"
#include "macroses.h"

// Copy constructor
CPermutohedral::CPermutohedral(const CPermutohedral &rhs)
    : m_nFeatures(rhs.m_nFeatures)
	, m_M(rhs.m_M)
	, m_featureSize(rhs.m_featureSize)
{
	if (!rhs.m_offset.empty()) rhs.m_offset.copyTo(m_offset); 
	if (!rhs.m_barycentric.empty()) rhs.m_barycentric.copyTo(m_barycentric);
	if (!rhs.m_blurNeighbor1.empty()) rhs.m_blurNeighbor1.copyTo(m_blurNeighbor1);
	if (!rhs.m_blurNeighbor2.empty()) rhs.m_blurNeighbor2.copyTo(m_blurNeighbor2);
}

// Copy operator
CPermutohedral & CPermutohedral::operator= (const CPermutohedral &rhs)
{
    if (&rhs == this) return *this;

	m_nFeatures		= rhs.m_nFeatures;
    m_M				= rhs.m_M;
    m_featureSize	= rhs.m_featureSize;
    
	m_offset		= rhs.m_offset.empty()			? Mat() : rhs.m_offset.clone();
	m_barycentric	= rhs.m_barycentric.empty()		? Mat() : rhs.m_barycentric.clone();
	m_blurNeighbor1 = rhs.m_blurNeighbor1.empty()	? Mat() : rhs.m_blurNeighbor1.clone();
	m_blurNeighbor2 = rhs.m_blurNeighbor2.empty()	? Mat() : rhs.m_blurNeighbor2.clone();

	return *this;
}

void CPermutohedral::init(const Mat &features)
{
    // Compute the lattice coordinates for each feature [there is going to be a lot of magic here
    m_nFeatures = features.rows;
    m_featureSize = features.cols;
    CHashTable hash_table(m_featureSize, m_nFeatures * (m_featureSize + 1));    // <============ Hash table
	//std::unordered_map<Mat, int> hash_table1;

    // Allocate the class memory
	m_offset		= Mat(m_nFeatures, m_featureSize + 1, CV_32SC1); 
    m_barycentric	= Mat(m_nFeatures, m_featureSize + 1, CV_32FC1);

    
    // Allocate the local memory
    vec_float_t scale_factor(m_featureSize);
    vec_float_t elevated(m_featureSize + 1);
    vec_float_t rem0(m_featureSize + 1);
    vec_float_t barycentric(m_featureSize + 2);
    std::vector<short> rank(m_featureSize + 1);
    std::vector<short> canonical((m_featureSize + 1) * (m_featureSize + 1));
	Mat key(1, m_featureSize + 1, CV_16SC1);
    
    // Compute the canonical simplex
    for(int i = 0; i <= m_featureSize; i++) {
        for(int j = 0; j <= m_featureSize - i; j++)
            canonical[i * (m_featureSize + 1) + j] = i;
        for(int j = m_featureSize - i + 1; j <= m_featureSize; j++)
            canonical[i * (m_featureSize + 1) + j] = i - (m_featureSize + 1);
    }
    
    // Expected standard deviation of our filter (p.6 in [Adams etal 2010])
    float inv_std_dev = sqrtf(2.f / 3.f) * (m_featureSize + 1);
    // Compute the diagonal part of E (p.5 in [Adams etal 2010])
    for(int i = 0; i < m_featureSize; i++)
        scale_factor[i] = 1.f / sqrtf((i + 2.f) * (i + 1.f)) * inv_std_dev;
    
    // Compute the simplex each feature lies in
    for(int k = 0; k < m_nFeatures; k++) {
        // Elevate the feature ( y = Ep, see p.5 in [Adams etal 2010])
        const float *f = features.ptr<float>(k);
        
        // sm contains the sum of 1..n of our faeture vector
        float sm = 0;
        for(int j = m_featureSize; j > 0; j--){
            float cf = f[j-1]*scale_factor[j-1];
            elevated[j] = sm - j*cf;
            sm += cf;
        }
        elevated[0] = sm;
        
        // Find the closest 0-colored simplex through rounding
        float down_factor = 1.0f / (m_featureSize + 1);
        float up_factor = static_cast<float>(m_featureSize + 1);
        int sum = 0;
        for(int i = 0; i <= m_featureSize; i++) {
            int rd = static_cast<int>(round( down_factor * elevated[i]));
            rem0[i] = rd*up_factor;
            sum += rd;
        }
        
        // Find the simplex we are in and store it in rank (where rank describes what position coorinate i has in the sorted order of the features values)
        for(int i = 0; i <= m_featureSize; i++)
            rank[i] = 0;
        for(int i = 0; i < m_featureSize; i++) {
            double di = elevated[i] - rem0[i];
            for(int j = i + 1; j <= m_featureSize; j++)
                if (di < elevated[j] - rem0[j])    rank[i]++;
                else                            rank[j]++;
        }
        
        // If the point doesn't lie on the plane (sum != 0) bring it back
        for(int i = 0; i <= m_featureSize; i++) {
            rank[i] += sum;
            if ( rank[i] < 0 ){
                rank[i] += m_featureSize + 1;
                rem0[i] += m_featureSize + 1;
            }
            else if (rank[i] > m_featureSize) {
                rank[i] -= m_featureSize + 1;
                rem0[i] -= m_featureSize + 1;
            }
        }
        
        // Compute the barycentric coordinates (p.10 in [Adams etal 2010])
        for(int i = 0; i <= m_featureSize + 1; i++)
            barycentric[i] = 0;
        for(int i = 0; i <= m_featureSize; i++) {
            float v = (elevated[i] - rem0[i])*down_factor;
            barycentric[m_featureSize - rank[i]  ] += v;
            barycentric[m_featureSize - rank[i] + 1] -= v;
        }
        // Wrap around
        barycentric[0] += 1.0f + barycentric[m_featureSize + 1];
        
        // Compute all vertices and their offset
		int		*pOffset		= m_offset.ptr<int>(k);
		float	*pBarycentric	= m_barycentric.ptr<float>(k);
		for(int remainder = 0; remainder <= m_featureSize; remainder++) {
            for(int i = 0; i < m_featureSize; i++)
                key.at<short>(0, i) = static_cast<short>(rem0[i] + canonical[ remainder * (m_featureSize + 1) + rank[i]]);		// TODO
			
			int val = hash_table.find(key);
			if (val == -1) {
				val = hash_table.size();
				hash_table.insert(key, val);
			}
			pOffset[remainder]		= val;	
			pBarycentric[remainder]	= barycentric[remainder];		
        }
    } // k
    
    // Find the Neighbors of each lattice point
    // Get the number of vertices in the lattice
    m_M = hash_table.size();
    
    // Create the neighborhood structure
	m_blurNeighbor1 = Mat(m_M, m_featureSize + 1, CV_32SC1);
	m_blurNeighbor2 = Mat(m_M, m_featureSize + 1, CV_32SC1);
    
    Mat n1(1, m_featureSize + 1, CV_16SC1);
    Mat n2(1, m_featureSize + 1, CV_16SC1);
    
    // For each of d+1 axes,
	for (int i = 0; i < m_M; i++) {
		int *pBlurNeighbor1 = m_blurNeighbor1.ptr<int>(i);
		int *pBlurNeighbor2 = m_blurNeighbor2.ptr<int>(i);
		
		Mat key = hash_table.getKey(i);

		for(int j = 0; j <= m_featureSize; j++) {

			for(int k = 0; k < m_featureSize; k++) {
                n1.at<short>(0, k) = key.at<short>(0, k) - 1;	// TODO
                n2.at<short>(0, k) = key.at<short>(0, k) + 1;	// TODO
            }
            n1.at<short>(0, j) = key.at<short>(0, std::min(j, m_featureSize - 1)) + m_featureSize;	// TODO
            n2.at<short>(0, j) = key.at<short>(0, std::min(j, m_featureSize - 1)) - m_featureSize;	// TODO
            
            pBlurNeighbor1[j] = hash_table.find(n1);
            pBlurNeighbor2[j] = hash_table.find(n2);
        }
    }
}

void CPermutohedral::compute(const Mat &src, Mat &dst, int in_offset, int out_offset, size_t in_size, size_t out_size) const
{
	if (in_size  == 0) in_size  = m_nFeatures - in_offset;
    if (out_size == 0) out_size = m_nFeatures - out_offset;
	if (dst.empty())   dst		= Mat(out_size, src.cols, CV_32FC1);

    // Shift all values by 1 such that -1 -> 0 (used for blurring)
	Mat values(m_M + 2, src.cols, CV_32FC1, Scalar(0));
    Mat newValues(m_M + 2, src.cols, CV_32FC1, Scalar(0));

    // Splatting
    for(int i = 0; i < in_size; i++) {
        const float *pIn			= src.ptr<float>(i);
		const int	*pOffset		= m_offset.ptr<int>(in_offset + i);
		const float	*pBarycentric	= m_barycentric.ptr<float>(in_offset + i);
		for(int j = 0; j <= m_featureSize; j++) {
            int   o	= pOffset[j] + 1;
            float w = pBarycentric[j];	
			float *pValues = values.ptr<float>(o);
			for(int k = 0; k < src.cols; k++)
				pValues[k] += w * pIn[k];
        }
    }
    
    for(int j = 0; j <= m_featureSize; j++) {
        for(int i = 0; i < m_M; i++) {
            float *pValues		= values.ptr<float>(i + 1);
            float *pNewValues	= newValues.ptr<float>(i + 1);
            
            int n1 = m_blurNeighbor1.at<int>(i, j) + 1;
            int n2 = m_blurNeighbor2.at<int>(i, j) + 1;
            float *n1_val = values.ptr<float>(n1);
            float *n2_val = values.ptr<float>(n2);
            for(int k = 0; k < src.cols; k++)
				pNewValues[k] = pValues[k] + 0.5f * (n1_val[k] + n2_val[k]);
        }
		swap(values, newValues);
    }
    // Alpha is a magic scaling constant (write Andrew if you really wanna understand this)
    float alpha = 1.0f / (1.0f + powf(2.0f, -static_cast<float>(m_featureSize)));
    
    // Slicing
    for(int i = 0; i < out_size; i++) {
        float		*pOut			= dst.ptr<float>(i);
		const int	*pOffset		= m_offset.ptr<int>(in_offset + i);
		const float	*pBarycentric	= m_barycentric.ptr<float>(in_offset + i);
		for(int k = 0; k < src.cols; k++)
            pOut[k] = 0;
        for(int j = 0; j <= m_featureSize; j++) {
            int   o = pOffset[j] + 1;		
            float w = pBarycentric[j];	
			float *pValues = values.ptr<float>(o);
			for(int k = 0; k < src.cols; k++)
                pOut[k] += w * pValues[k] * alpha;
        }
    }
}
