/// @cond DemoCode
#pragma once

#include "DGM.h"

using namespace DirectGraphicalModels;

class CDemo1D
{
public:
    virtual ~CDemo1D(void) {}
	virtual void Main(void) = 0;
};
/// @endcond

/**
@page demo1d Demo 1D
Here we introduce the application of the DGM library to some well-known probabilistic models. Particaulary, we show how to compose graphical models with the help of 
DirectGraphicalModels::CGraphPairwise class. We fill the model by hand with potentials and study inferece and decoding processes on them. First we demonstrate brute-force 
algorithms for exact decoding and inference on a small graph, and consider a possible use of marginal probabilities calculated by inference algorithm for approximate 
decoding. Next we show message-passing algorithms for approximate inference, and consider the difference between \a sum-product and \a max-sum approaches. Finally, 
we consider \a sum-product message-passing algorithms, which provide an efficient framework for exact inference in chain- and tree-structuresd graphs.

  - @subpage demo1d_exact : An introduction to graphical models and the tasks of decoding and inference on a small graphical model where we can do everything by hand. 
  - @subpage demo1d_chain : An introduction to Markov independence properties on an example of a chain-structured graphical model, and to efficient dynamic programming 
							algorithms for inference. 
  - @subpage demo1d_tree : This demo shows how to construct a tree-structured graphical model, for which also an exact message-passing inference algorithm exists. 
*/
