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

// Copy constructor
Permutohedral::Permutohedral(const Permutohedral &o) : m_pOffset(NULL), m_pBarycentric(NULL), m_pBlurNeighbors(NULL), m_N(o.m_N), m_M(o.m_M), m_d(o.m_d)
{
    if (o.m_pBarycentric) {
        m_pBarycentric = new float[(m_d + 1) * m_N];
        memcpy(m_pBarycentric, o.m_pBarycentric, (m_d + 1) * m_N * sizeof(float));
    }
    if (o.m_pOffset) {
        m_pOffset = new int[(m_d + 1) * m_N];
        memcpy(m_pOffset, o.m_pOffset, (m_d + 1) * m_N * sizeof(int));
    }
    if (o.m_pBlurNeighbors) {
        m_pBlurNeighbors = new Neighbors[(m_d + 1) * m_N];
        memcpy(m_pBlurNeighbors, o.m_pBlurNeighbors, (m_d + 1) * m_N * sizeof(Neighbors));
    }
}

// Copy operator
Permutohedral & Permutohedral::operator= (const Permutohedral &o)
{
    if (&o == this) return *this;
    if (m_pBarycentric)    delete[] m_pBarycentric;
    if (m_pOffset)         delete[] m_pOffset;
    if (m_pBlurNeighbors)  delete[] m_pBlurNeighbors;
    
	m_pOffset			= NULL; 
	m_pBarycentric		= NULL; 
	m_pBlurNeighbors	= NULL;
    m_N = o.m_N;
    m_M = o.m_M;
    m_d = o.m_d;
    
	if (o.m_pBarycentric){
        m_pBarycentric = new float[(m_d + 1) * m_N];
        memcpy( m_pBarycentric, o.m_pBarycentric, (m_d + 1) * m_N * sizeof(float));
    }
    
	if (o.m_pOffset){
        m_pOffset = new int[(m_d + 1) * m_N];
        memcpy(m_pOffset, o.m_pOffset, (m_d + 1) * m_N * sizeof(int));
    }
    
	if (o.m_pBlurNeighbors){
        m_pBlurNeighbors = new Neighbors[(m_d + 1) * m_N];
        memcpy(m_pBlurNeighbors, o.m_pBlurNeighbors, (m_d + 1) * m_N * sizeof(Neighbors));
    }
    
	return *this;
}

// Destructor
Permutohedral::~Permutohedral(void)
{
    if (m_pBarycentric)    delete[] m_pBarycentric;
    if (m_pOffset)         delete[] m_pOffset;
    if (m_pBlurNeighbors)  delete[] m_pBlurNeighbors;
}

void Permutohedral::init(const float * feature, int feature_size, int N)
{
    // Compute the lattice coordinates for each feature [there is going to be a lot of magic here
    m_N = N;
    m_d = feature_size;
    CHashTable hash_table(m_d, m_N * (m_d + 1));    // <============ Hash table
	//std::unordered_map<short, int> hash_table;

    // Allocate the class memory
    if (m_pOffset) delete [] m_pOffset;
    m_pOffset = new int[(m_d + 1) * m_N];
    if (m_pBarycentric) delete [] m_pBarycentric;
    m_pBarycentric = new float[(m_d + 1) * m_N];
    
    // Allocate the local memory
    float * scale_factor = new float[m_d];
    float * elevated = new float[m_d + 1];
    float * rem0 = new float[m_d + 1];
    float * barycentric = new float[m_d + 2];
    short * rank = new short[m_d + 1];
    short * canonical = new short[(m_d + 1) * (m_d + 1)];
    short * key = new short[m_d + 1];
    
    // Compute the canonical simplex
    for(int i = 0; i <= m_d; i++) {
        for(int j = 0; j <= m_d - i; j++)
            canonical[i * (m_d + 1) + j] = i;
        for(int j = m_d - i + 1; j <= m_d; j++)
            canonical[i * (m_d + 1) + j] = i - (m_d + 1);
    }
    
    // Expected standard deviation of our filter (p.6 in [Adams etal 2010])
    float inv_std_dev = sqrtf(2.f / 3.f) * (m_d + 1);
    // Compute the diagonal part of E (p.5 in [Adams etal 2010])
    for(int i = 0; i < m_d; i++)
        scale_factor[i] = 1.f / sqrtf((i + 2.f) * (i + 1.f)) * inv_std_dev;
    
    // Compute the simplex each feature lies in
    for(int k = 0; k < m_N; k++) {
        // Elevate the feature ( y = Ep, see p.5 in [Adams etal 2010])
        const float * f = feature + k*feature_size;
        
        // sm contains the sum of 1..n of our faeture vector
        float sm = 0;
        for(int j = m_d; j > 0; j--){
            float cf = f[j-1]*scale_factor[j-1];
            elevated[j] = sm - j*cf;
            sm += cf;
        }
        elevated[0] = sm;
        
        // Find the closest 0-colored simplex through rounding
        float down_factor = 1.0f / (m_d + 1);
        float up_factor = (m_d + 1);
        int sum = 0;
        for(int i = 0; i <= m_d; i++) {
            int rd = static_cast<int>(round( down_factor * elevated[i]));
            rem0[i] = rd*up_factor;
            sum += rd;
        }
        
        // Find the simplex we are in and store it in rank (where rank describes what position coorinate i has in the sorted order of the features values)
        for(int i = 0; i <= m_d; i++)
            rank[i] = 0;
        for(int i = 0; i < m_d; i++) {
            double di = elevated[i] - rem0[i];
            for(int j = i + 1; j <= m_d; j++)
                if (di < elevated[j] - rem0[j])    rank[i]++;
                else                            rank[j]++;
        }
        
        // If the point doesn't lie on the plane (sum != 0) bring it back
        for(int i = 0; i <= m_d; i++) {
            rank[i] += sum;
            if ( rank[i] < 0 ){
                rank[i] += m_d + 1;
                rem0[i] += m_d + 1;
            }
            else if (rank[i] > m_d) {
                rank[i] -= m_d + 1;
                rem0[i] -= m_d + 1;
            }
        }
        
        // Compute the barycentric coordinates (p.10 in [Adams etal 2010])
        for(int i = 0; i <= m_d + 1; i++)
            barycentric[i] = 0;
        for(int i = 0; i <= m_d; i++) {
            float v = (elevated[i] - rem0[i])*down_factor;
            barycentric[m_d - rank[i]  ] += v;
            barycentric[m_d - rank[i] + 1] -= v;
        }
        // Wrap around
        barycentric[0] += 1.0f + barycentric[m_d + 1];
        
        // Compute all vertices and their offset
        for(int remainder = 0; remainder <= m_d; remainder++) {
            for(int i = 0; i < m_d; i++)
                key[i] = rem0[i] + canonical[ remainder * (m_d + 1) + rank[i]];
            m_pOffset[k * (m_d + 1) + remainder] = hash_table.find(key, true);
            m_pBarycentric[k * (m_d + 1) + remainder] = barycentric[remainder];
        }
    }
    delete [] scale_factor;
    delete [] elevated;
    delete [] rem0;
    delete [] barycentric;
    delete [] rank;
    delete [] canonical;
    delete [] key;
    
    
    // Find the Neighbors of each lattice point
    
    // Get the number of vertices in the lattice
    m_M = hash_table.size();
    
    // Create the neighborhood structure
    if(m_pBlurNeighbors) delete[] m_pBlurNeighbors;
    m_pBlurNeighbors = new Neighbors[(m_d + 1) * m_M];
    
    short * n1 = new short[m_d + 1];
    short * n2 = new short[m_d + 1];
    
    // For each of d+1 axes,
    for(int j = 0; j <= m_d; j++) {
        for(int i = 0; i< m_M; i++) {
            const short * key = hash_table.getKey( i );
            for(int k = 0; k < m_d; k++) {
                n1[k] = key[k] - 1;
                n2[k] = key[k] + 1;
            }
            n1[j] = key[j] + m_d;
            n2[j] = key[j] - m_d;
            
            m_pBlurNeighbors[j * m_M + i].n1 = hash_table.find(n1);
            m_pBlurNeighbors[j * m_M + i].n2 = hash_table.find(n2);
        }
    }
    
	delete[] n1;
    delete[] n2;
}

void Permutohedral::compute(float * out, const float * in, int value_size, int in_offset, int out_offset, int in_size, int out_size) const
{
    if (in_size  == -1) in_size  = m_N - in_offset;
    if (out_size == -1) out_size = m_N - out_offset;
    
    // Shift all values by 1 such that -1 -> 0 (used for blurring)
    float * values = new float[(m_M + 2) * value_size];
    float * new_values = new float[(m_M + 2) * value_size];
    
    for(int i = 0; i < (m_M + 2) * value_size; i++)
        values[i] = new_values[i] = 0;
    
    // Splatting
    for(int i = 0; i < in_size; i++) {
        for(int j = 0; j <= m_d; j++) {
            int o = m_pOffset[(in_offset + i) * (m_d + 1) + j] + 1;
            float w = m_pBarycentric[(in_offset + i) * (m_d + 1) + j];
            for(int k = 0; k < value_size; k++)
                values[o * value_size + k] += w * in[i * value_size + k];
        }
    }
    
    for(int j = 0; j <= m_d; j++) {
        for(int i = 0; i < m_M; i++) {
            float * old_val = values + (i + 1) * value_size;
            float * new_val = new_values + (i + 1) * value_size;
            
            int n1 = m_pBlurNeighbors[j * m_M + i].n1 + 1;
            int n2 = m_pBlurNeighbors[j * m_M + i].n2 + 1;
            float * n1_val = values + n1 * value_size;
            float * n2_val = values + n2 * value_size;
            for(int k = 0; k < value_size; k++)
                new_val[k] = old_val[k] + 0.5f * (n1_val[k] + n2_val[k]);
        }
        float * tmp = values;
        values = new_values;
        new_values = tmp;
    }
    // Alpha is a magic scaling constant (write Andrew if you really wanna understand this)
    float alpha = 1.0f / (1.0f + powf(2.0f, -static_cast<float>(m_d)));
    
    // Slicing
    for(int i = 0; i < out_size; i++) {
        for(int k = 0; k < value_size; k++)
            out[i * value_size + k] = 0;
        for(int j = 0; j <= m_d; j++) {
            int o = m_pOffset[(out_offset + i) * (m_d + 1) + j] + 1;
            float w = m_pBarycentric[(out_offset + i) * (m_d + 1) + j];
            for(int k = 0; k < value_size; k++)
                out[i * value_size + k] += w * values[o * value_size + k] * alpha;
        }
    }
    
    
    delete[] values;
    delete[] new_values;
}
