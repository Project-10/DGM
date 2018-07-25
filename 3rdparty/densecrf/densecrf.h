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
#include "permutohedral.h"
#include "pairwisepotential.h"

class PairwisePotential;

class DenseCRF {
protected:
	friend class BipartiteDenseCRF;
	
	// Number of variables and labels
	int N_, M_;
	float *unary_, *additional_unary_, *current_, *next_, *tmp_;
	
	// Store all pairwise potentials
	std::vector<PairwisePotential *> pairwise_;
	
	// Run inference and return the pointer to the result
	float* runInference( int n_iterations, float relax);
	
	// Auxillary functions
	void expAndNormalize( float* out, const float* in, float scale = 1.0, float relax = 1.0 );
	
	// Don't copy this object, bad stuff will happen
	DenseCRF( DenseCRF & o ){}

public:
	// Create a dense CRF model of size N with M labels
	DllExport DenseCRF( int N, int M );
	DllExport virtual ~DenseCRF();
	// Add  a pairwise potential defined over some feature space
	// The potential will have the form:    w*exp(-0.5*|f_i - f_j|^2)
	// The kernel shape should be captured by transforming the
	// features before passing them into this function
	DllExport void addPairwiseEnergy( const float * features, int D, float w=1.0f, const SemiMetricFunction * function=NULL );
	
	// Add your own favorite pairwise potential (ownwership will be transfered to this class)
	DllExport void addPairwiseEnergy( PairwisePotential* potential );
	
	// Set the unary potential for all variables and labels (memory order is [x0l0 x0l1 x0l2 .. x1l0 x1l1 ...])
	DllExport void setUnaryEnergy( const float * unary );
	
	// Set the unary potential for a specific variable
	DllExport void setUnaryEnergy( int n, const float * unary );
	
	// Run inference and return the probabilities
	DllExport void inference( int n_iterations, float* result, float relax=1.0 );
	
	// Run MAP inference and return the map for each pixel
	DllExport void map( int n_iterations, short int* result, float relax=1.0 );
	
	// Step by step inference
	DllExport void startInference();
	DllExport void stepInference( float relax = 1.0 );
	DllExport void currentMap( short * result );
	
public: /* Debugging functions */
	// Compute the unary energy of an assignment
	DllExport void unaryEnergy( const short * ass, float * result );
	
	// Compute the pairwise energy of an assignment (half of each pairwise potential is added to each of it's endpoints)
	DllExport void pairwiseEnergy( const short * ass, float * result, int term=-1 );
};


class DenseCRF2D : public DenseCRF {

protected:
	// Width, height of the 2d grid
	int W_, H_;

public:
	// Create a 2d dense CRF model of size W x H with M labels
	DllExport DenseCRF2D( int W, int H, int M );
	DllExport virtual ~DenseCRF2D();
	// Add a Gaussian pairwise potential with standard deviation sx and sy
	DllExport void addPairwiseGaussian( float sx, float sy, float w, const SemiMetricFunction * function = NULL );
	
	// Add a Bilateral pairwise potential with spacial standard deviations sx, sy and color standard deviations sr,sg,sb
	DllExport void addPairwiseBilateral( float sx, float sy, float sr, float sg, float sb, const unsigned char * im, float w, const SemiMetricFunction * function = NULL );
	
	// Set the unary potential for a specific variable
	DllExport void setUnaryEnergy( int x, int y, const float * unary );
	using DenseCRF::setUnaryEnergy;
};

