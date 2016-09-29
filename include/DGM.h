#pragma once

#include "DGM/CMat.h"
#include "DGM/FeaturesConcatenator.h"
#include "DGM/NDGauss.h"
#include "DGM/random.h"
#include "DGM/parallel.h"

#include "DGM/IPDF.h"
#include "DGM/PDFHistogram.h"
#include "DGM/PDFGaussian.h"

#include "DGM/Prior.h"
#include "DGM/PriorNode.h"
#include "DGM/PriorEdge.h"
#include "DGM/PriorTriplet.h"

#include "DGM/ITrain.h"
#include "DGM/TrainNode.h"
#include "DGM/TrainNodeNaiveBayes.h"
#include "DGM/TrainNodeGM.h"
#include "DGM/TrainNodeGMM.h"
#include "DGM/TrainNodeCvGMM.h"
#include "DGM/TrainNodeCvGM.h"
#include "DGM/TrainNodeCvRF.h"
#include "DGM/TrainNodeMsRF.h"
#include "DGM/TrainEdge.h"
#include "DGM/TrainEdgePotts.h"
#include "DGM/TrainEdgePottsCS.h"
#include "DGM/TrainEdgePrior.h"
#include "DGM/TrainEdgeConcat.h"
#include "DGM/TrainTriplet.h"
#include "DGM/TrainLink.h"
#include "DGM/TrainLinkNested.h"

#include "DGM/IGraph.h"
#include "DGM/Graph.h"
#include "DGM/GraphWeiss.h"
#include "DGM/GraphExt.h"
#include "DGM/Graph3.h"

#include "DGM/Infer.h"
#include "DGM/InferExact.h"
#include "DGM/InferChain.h"
#include "DGM/InferTree.h"
#include "DGM/InferLBP.h"
#include "DGM/InferTRW.h"
#include "DGM/InferViterbi.h"

#include "DGM/Decode.h"
#include "DGM/DecodeExact.h"

#include "DGM/Powell.h"

/**
@mainpage Introduction
@section sec_main Direct Graphical Models (DGM)
is a C++ dynamic link library implementing various tasks in <a href="https://en.wikipedia.org/wiki/Graphical_model">probabilistic graphical models</a> with pairwise dependencies. 
The library aims to be used for the <a href="https://en.wikipedia.org/wiki/Markov_random_field">Markov-</a> and 
<a href="https://en.wikipedia.org/wiki/Conditional_random_field">Conditional Random Fields</a> (MRF / CRF), 
<a href="https://en.wikipedia.org/wiki/Markov_chain">Markov Chains</a>, <a href="https://en.wikipedia.org/?title=Bayesian_network">Bayesian Networks</a>, @a etc. 
DGM library consists of three modules: 
- Main @ref moduleDGM, which includes a variety of methods for the tasks:
	- @ref sec_main_train
	- @ref moduleGraph
	- @ref sec_main_decode
	- @ref sec_main_paramest
	- @ref moduleEva
- Feature extraction @ref moduleFEX, which allows for extracting the main data features, used nowadays in image classification.
- Visualization @ref moduleVIS, wich allows for visualizing the used data and features as well as intermediate and final classification results.

These tasks are optimized for speed, @a i.e. high-efficient calculations. The code is written entirely in C++ and can be compiled with Microsoft Visual C++.

@section sec_main_overview Methods Overview
@subsection sec_main_train Training
DGM implements the following training methods: 

@subsubsection sec_main_train_nodes Nodes | Unary Potentials
- <b>NaiveBayes:</b> Naive Bayes training @ref DirectGraphicalModels::CTrainNodeNaiveBayes
- <b>GM:</b> Gaussian Model training @ref DirectGraphicalModels::CTrainNodeGM
- <b>CvGM:</b> OpenCV Gaussian Model training @ref DirectGraphicalModels::CTrainNodeCvGM
- <b>GMM:</b> Gaussian Mixture Model training (<a href="http://www.project-10.de/Kosov/files/GCPR_2013.pdf" target="_blank">Sequential GMM Training Algorithm</a>) @ref DirectGraphicalModels::CTrainNodeGMM
- <b>CvGMM:</b> OpenCV Gaussian Mixture Model training @ref DirectGraphicalModels::CTrainNodeCvGMM
- <b>CvRF:</b> OpenCV Random Forest training @ref DirectGraphicalModels::CTrainNodeCvRF
- <b>MsRF:</b> Microsoft Research Random Forest training @ref DirectGraphicalModels::CTrainNodeMsRF

The corresponding classes are @b CTrainNode* (where @b * is the name of the method above). The difference between these methods is described at forum:
<a href="http://www.project-10.de/forum/viewtopic.php?f=22&t=954">Training of a Random Model</a>.

@subsubsection sec_main_train_edges Edges | Pairwise Potentials
- <b>Potts:</b> Train- & Test-data-independent Potts model @ref DirectGraphicalModels::CTrainEdgePotts
- <b>PottsCS:</b> Train-data-independent contrast-sensitive Potts model @ref DirectGraphicalModels::CTrainEdgePottsCS
- <b>Prior:</b> Contrast-sensitive Potts model with prior probability @ref DirectGraphicalModels::CTrainEdgePrior
- <b>Concat:</b> Concatenated training @ref DirectGraphicalModels::CTrainEdgeConcat

The corresponding classes are @b CTrainEdge* (where @b * is the name of the method above).

@subsection sec_main_decode Inference / Decode
DGM implements the following inference and decoding methods: 

@subsubsection sec_main_decode_inference Inference
- <b>Exact:</b> Exact inferece for small graphs with an exhaustive search @ref DirectGraphicalModels::CInferExact
- <b>Chain:</b> Exact inferece for Markov chains (chain-structured graphs) @ref DirectGraphicalModels::CInferChain
- <b>Tree:</b> Exact inferece for undirected graphs without loops (tree-structured graphs) @ref DirectGraphicalModels::CInferTree
- <b>LBP:</b> Approximate inference based on the Loopy Belief Propagation (\a sum-product message-passing) algorithm @ref DirectGraphicalModels::CInferLBP 
- <b>TRW:</b> Approximate inference based on the (<a href="http://pub.ist.ac.at/~vnk/papers/TRW-S-PAMI.pdf" target="_blank">Convergent Tree-Reweighted</a>) (\a max-sum message-passing) algorithm @ref DirectGraphicalModels::CInferTRW 
- <b>Viterbi:</b> Approximate inference based on Viterbi (\a max-sum message-passing) algorithm @ref DirectGraphicalModels::CInferViterbi 

The corresponding classes are @b CInfer* (where @b * is the name of the method above). 

All of the inference classes may be also used for approximate decoding via function @ref DirectGraphicalModels::CInfer::decode()

@subsubsection sec_main_decode_decoding Decoding
- <b>Exact:</b> Exact decoding for small graphs with an exhaustive search @ref DirectGraphicalModels::CDecodeExact

The corresponding classes are @b CDecode* (where @b * is the name of the method above). 

@subsection sec_main_paramest Parameter Estimation
DGM implements the following parameter estimation method:
- <b>Powell:</b> Powell search method @ref DirectGraphicalModels::CPowell

@subsection sec_main_sampling Sampling
DGM implements the following sampling method:
- <b>Gauss:</b> Sampling from the multivariate gaussian distribution @ref DirectGraphicalModels::CNDGauss::getSample()

@subsection sec_main_fex Feature Extraction
Please refer to the @ref moduleFEX documentation

@subsection sec_main_marker Visualization
Please refer to the @ref moduleVIS documentation


@section sec_main_links Quick Links
- @ref s3
- @ref demo
- <a href="http://project-10.de/forum/viewforum.php?f=31"><b>User Q&A forum</b></a>
- <a href="https://github.com/Project-10/DGM/issues"><b>Report a bug</b></a>


@author Sergey G. Kosov, sergey.kosov@project-10.de
*/

/**
@defgroup moduleDGM DGM Module
@section sec_dgm_main Main DGM Module

	@defgroup moduleTrain Training
	@ingroup moduleDGM
	
		@defgroup moduleTrainNode Unary Potentials Training
		@ingroup moduleTrain

		@defgroup moduleTrainEdge Pairwise Potentials Training
		@ingroup moduleTrain

	@defgroup moduleGraph Graph Building
	@ingroup moduleDGM

	@defgroup moduleDecode Inference / Decoding
	@ingroup moduleDGM

	@defgroup moduleParamEst Parameter Estimation
	@ingroup moduleDGM

	@defgroup moduleEva Evaluation
	@ingroup moduleDGM
*/

/**
@page s3 Installation
Curently the DGM library may be installed only on Windows machines. It is also based on OpenCV library v.3.1.0.<br> 
In order to build the DGM library, the OpenCV library should be built and installed first.

@section sec_install_cv Installing OpenCV
-# Download the OpenCV library from <a href="http://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.1.0/" target="_blank">sourcefourge</a>
-# Install the OpenCV library. You may follow the <a href="http://www.project-10.de/forum/viewtopic.php?f=23&t=198#p237" target="_blank">short installation guide</a> or a deteiled
   <a href="http://docs.opencv.org/2.4/doc/tutorials/introduction/windows_install/windows_install.html#installation-by-making-your-own-libraries-from-the-source-files" target="_blank">installation in Windows guide</a>

@section sec_install_dgm Installing DGM
-# Download the DGM library from <a href="http://research.project-10.de/dgm/" target="_blank">Project X Research</a>
-# Unzip it to your local folder (for example to disk @b C:\\, so the library path will be @b C:\\DGM\\)
-# In case you want to build the library follow these instructions, otherwise - skip this step
	-# Download and install <a href="https://cmake.org/" target="_blank">CMake</a>
    -# Run \b cmake-gui.exe
    -# In the <i>“Where is the source code”</i> field choose the DGM source directory: \b C:\\DGM\\sources<br>
       In the <i>“Where to build the binaries”</i> field choose directory for VS compiled DGM: \b C:\\DGM\\builds
    -# Press \a Configure button and choose <i>Visual Studio 14 2015</i> or <i>Visual Studio 14 2015 Win64</i>  (or whatever) as building environment
    -# Be sure that the \a OpenCV_DIR is pointing to the OpenCV build directory (\a e.g. \b C:\\OpenCV\\build), where \b OpenCVConfig.cmake file is located
    -# Press one more time \a Configure and then \a Generate, so the VS project will be generated in the \b C:\\DGM\\builds
    -# Open the solution (\b DGM.sln file) 
    -# Build \b ALL_BUILD and \b INSTALL projects first for \a Debug and then for \a Release configuration. That will copy DGM headers, binaries and demonstration applications to \b C:\\DGM\\builds\\install
    -# (Optionally) you can copy the content of the \b C:\\DGM\\builds\\install to another folder, e.g. \b C:\\DGM\\build 
-# Specify the following paths and library
	-# Add to Configuration Properties -> C/C++ -> General -> Additional Include Directories the path \b C:\\DGM\\build\\include
	-# Add to Configuration Properties -> Linker -> General -> Additional Library Directories the path \b C:\\DGM\\build\\lib for both Release and Debug configurations
	-# Add to Configuration Properties -> Linker -> Input -> Additional Dependencies the libraries \b dgm150.lib, \b fex150.lib and \b dgm150d.lib, \b fex150d.lib for Release and Debug configurations accordingly
-# Copy the DGM dll files \b dgm150.dll, \b dgm150d.dll and \b fex150.dll, \b fex150d.dll and \b vis150.dll, \b vis150d.dll from @b C:\\DGM\\build\\bin to your project's Relese and Debug folders.
*/

/**
@page demo Tutorials
<h2>How to use the code</h2>
The documentation for DGM consists of a series of demos, showing how to use DGM to perform various tasks. These demos also contain some tutorial material on graphical models.

- @subpage demo1d : An introduction to graphical models and to the tasks of inference and decoding on a set of simple examples:
  - @ref demo1d_exact : An introduction to graphical models and the tasks of decoding and inference on a small graphical model where we can do everything by hand. 
  - @ref demo1d_chain : An introduction to Markov independence properties on an example of a chain-structured graphical model, and to efficient dynamic programming 
						algorithms for inference. 
  - @ref demo1d_tree : This demo shows how to construct a tree-structured graphical model, for which also an exact message-passing inference algorithm exists. 
- @subpage demo2d : An example of more complicated graphical models, containing loops and built upon a binary 2-dimentional image. This example also shows the application of DGM to
					unsupervised segmentation.
- @subpage demostereo : An example of CRFs application to the problem of disparity estimation between a pair of stereo images.
- @subpage demofex : An introduction to the feature extraction, needed mainly for supervised learning.
- @subpage demotrain : An introdiction to the random model learning (training) in case when the training data is available.
- @subpage demovis : An example of usage the visualization module of the library for analysis and represention of the intermediate and final results of classification.

*/

/**
@page demo2d Demo 2D
In this demo, we consider the case of binary variables with attractive potentials. The original binary image \b Smile.png is degraded with noise, 
as showed at image \b Smile_noise.png. Using the DGM we restore the original image from its noised version and evaluate the results by
comparing the restored image with the original one.

<table align="center">
<tr>
	<td colspan="2"><center><b>Input</b></center></td>
	<td></td>
	<td><center><b>Output</b></center></td>
</tr>
<tr>
	<td><img src="smile.png"></td>
	<td><img src="smile_noise.png"></td>
	<td><img src="arrow.png"></td>
	<td><img src="smile_denoised.png"></td>
</tr>
<tr>
	<td><center><b>Smile.png</b></center></td>
	<td><center><b>Smile_noise.png</b></center></td>
	<td></td>
	<td><center><b>Restored Image</b></center></td>
</tr>
</table>

This example copies the idea from the <a href="http://www.cs.ubc.ca/~schmidtm/Software/UGM/graphCuts.html">GraphCuts UGM Demo</a>
@code
#include "DGM.h"
using namespace DirectGraphicalModels;

int main(int argv, char *argc[]) 
{
	const unsigned int	nStates	= 2;						// {true; false}
	
	// Reading parameters and images
	Mat		  img		= imread("Smile.png", 0);			// original image
	Mat		  noise		= imread("Smile_noise.png", 0);		// noised image
	int		  width		= img.cols;
	int		  height	= img.rows;
	
	CGraph	* graph		= new CGraph(nStates);
	CInfer  * decoder	= new CInferViterbi(graph);

	Mat nodePot(nStates, 1, CV_32FC1);						// node Potential (column-vector)
	Mat edgePot(nStates, nStates, CV_32FC1);				// edge Potential	
	
	// No training
	// Defynig the edge potential
	edgePot = CTrainEdgePotts::getEdgePotentials(10000, nStates);
	// equivalent to:
	// ePot.at<float>(0, 0) = 1000;	ePot.at<float>(0, 1) = 1;
	// ePot.at<float>(1, 0) = 1;	ePot.at<float>(1, 1) = 1000;

	// ==================== Building and filling the graph ====================
	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++) {
			float p = 1.0f - static_cast<float>(noise.at<byte>(y,x)) / 255.0f;
			nodePot.at<float>(0, 0) = p;
			nodePot.at<float>(1, 0) = 1.0f - p;
			size_t idx = graph->addNode(nodePot);
			if (x > 0) graph->addArc(idx, idx - 1, edgePot);
			if (y > 0) graph->addArc(idx, idx - width, edgePot);
			if ((x > 0) && (y > 0)) graph->addArc(idx, idx - width - 1, edgePot);
			if ((x < width - 1) && (y > 0)) graph->addArc(idx, idx - width + 1, edgePot);									
		} // x

	// =============================== Decoding ===============================
	printf("Decoding... ");
	int64 ticks = getTickCount();
	std::vector<byte> optimalDecoding = decoder->decode(100);
	ticks =  getTickCount() - ticks;
	printf("Done! (%fms)\n", ticks * 1000 / getTickFrequency());

	
	// ====================== Evaluation / Visualization ======================
	noise = Mat(noise.size(), CV_8UC1, optimalDecoding.data()) * 255;
	medianBlur(noise, noise, 3);

	float error = 0;
	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
			if (noise.at<byte>(y,x) != img.at<byte>(y,x)) error++;

	printf("Accuracy  = %.2f%%\n", 100 - 100 * error / (width * height));
	
	imshow("Image", noise);	

	cvWaitKey();

	return 0;
}
@endcode
*/

/**
@page demotrain Demo Train
In this demo, we consider the case when the training data is aviable. In this example the trainig data is represented in form of 
manually labelled image. The original image \b Original \b Image.jpg is a color-infrared image, and the grounftruth image \b GroundTruth \b Image.jpg
has 6 different classes, namely \a road, \a traffic \a island, \a grass, \a agriculture, \a tree and \a car (instances of which are not represented in image). 

In this example DGM uses 3 features extacted from the \b Original \b Image.jpg for training (Please refer to the @ref demofex for details). The same image then is used for the labelling. Finally, 
we evaluate the results by comparing the lebelled image with the groundtruth.


<table align="center">
<tr>
	<td colspan="3"><center><b>Input</b></center></td>
	<td></td>
	<td><center><b>Output</b></center></td>
</tr>
<tr>
  <td><img src="001_img_small.jpg"></td>
  <td><img src="001_fv_small.jpg"></td>
  <td><img src="001_gt_small.jpg"></td>
  <td><img src="arrow.png"></td>
  <td><img src="001_res_small.jpg"></td>
</tr>
<tr>
  <td><center><b>Original Image.jpg</b></center></td>
  <td><center><b>Feature Vector.jpg</b></center></td>
  <td><center><b>GroundTruth Image.jpg</b></center></td>
  <td></td>
  <td><center><b>Resulting Class Map</b></center></td>
</tr>
</table>

@code
#include "DGM.h"
#include "VIS.h"
using namespace DirectGraphicalModels;
using namespace DirectGraphicalModels::vis;

int main(int argv, char *argc[])
{
	const CvSize		imgSize		= cvSize(400, 400);
	const int			width		= imgSize.width;
	const int			height		= imgSize.height;
	const unsigned int	nStates		= 6;	// {road, traffic island, grass, agriculture, tree, car} 		
	const unsigned int	nFeatures	= 3;		

	if (argv != 7) {
		print_help();
		return 0;
	}

	// Reading parameters and images
	int nodeModel	= atoi(argc[1]);															// node training model
	int edgeModel	= atoi(argc[2]);															// edge training model
	Mat img			= imread(argc[3], 1); resize(img, img, imgSize, 0, 0, INTER_LANCZOS4);		// image
	Mat fv			= imread(argc[4], 1); resize(fv,  fv,  imgSize, 0, 0, INTER_LANCZOS4);		// feature vector
	Mat gt			= imread(argc[5], 0); resize(gt,  gt,  imgSize, 0, 0, INTER_NEAREST);		// groundtruth

	CTrainNode		* nodeTrainer	= NULL; 
	CTrainEdge		* edgeTrainer	= NULL;
	CGraph			* graph			= new CGraph(nStates); 
	CInfer			* decoder		= new CInferLBP(graph);
	CMarker			* marker		= new CMarker(DEF_PALETTE_6);
	CCMat			* confMat		= new CCMat(nStates);
	float			  params[]		= {100, 0.01f};						
	size_t			  params_len;

	switch(nodeModel) {
		case 0: nodeTrainer = new CTrainNodeNaiveBayes(nStates, nFeatures);	break;
		case 1: nodeTrainer = new CTrainNodeGM(nStates, nFeatures);			break;
		case 2: nodeTrainer = new CTrainNodeGMM(nStates, nFeatures);		break;		
		case 3: nodeTrainer = new CTrainNodeCvGM(nStates, nFeatures);		break;
		case 4: nodeTrainer = new CTrainNodeCvGMM(nStates, nFeatures);		break;		
		case 5: nodeTrainer = new CTrainNodeCvRF(nStates, nFeatures);		break;		
		case 6: nodeTrainer = new CTrainNodeMsRF(nStates, nFeatures);		break;		
	}
	switch(edgeModel) {
		case 0: params[0] = 1;	// Emulate "No edges"
		case 1:	edgeTrainer = new CTrainEdgePotts(nStates, nFeatures);		params_len = 1; break;
		case 2:	edgeTrainer = new CTrainEdgePottsCS(nStates, nFeatures);	params_len = 2; break;
		case 3:	edgeTrainer = new CTrainEdgePrior(nStates, nFeatures);		params_len = 2; break;
		case 4:	
			CFeaturesConcatenator *pConcatenator = new CDiffFeaturesConcatenator(nFeatures);
			edgeTrainer = new CTrainEdgeConcat(nStates, nFeatures, pConcatenator);		
			params_len = 1;
			break;
	}

	// ==================== STAGE 1: Building the graph ====================
	printf("Building the Graph... ");
	int64 ticks = getTickCount();	
	for (int y = 0; y < height; y++) 
		for (int x = 0; x < width; x++) {
			size_t idx = graph->addNode();
			if (x > 0) 	 graph->addArc(idx, idx - 1);
			if (y > 0) 	 graph->addArc(idx, idx - width); 
		} // x
	ticks = getTickCount() - ticks;
	printf("Done! (%fms)\n", ticks * 1000 / getTickFrequency());

	// ========================= STAGE 2: Training =========================
	printf("Training... ");
	ticks = getTickCount();	
	
	// Node Training (copact notation)
	nodeTrainer->addFeatureVec(fv, gt);					
	nodeTrainer->train();	

	// Edge Training (comprehensive notation)
	Mat featureVector1(nFeatures, 1, CV_8UC1); 
	Mat featureVector2(nFeatures, 1, CV_8UC1); 	
	for (int y = 1; y < height; y++) {
		byte *pFv1 = fv.ptr<byte>(y);
		byte *pFv2 = fv.ptr<byte>(y - 1);
		byte *pGt1 = gt.ptr<byte>(y);
		byte *pGt2 = gt.ptr<byte>(y - 1);
		for (int x = 1; x < width; x++) {
			for (word f = 0; f < nFeatures; f++) featureVector1.at<byte>(f, 0) = pFv1[nFeatures * x + f];		// featureVector1 = fv[x][y]

			for (word f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pFv1[nFeatures * (x - 1) + f];	// featureVector2 = fv[x-1][y]
			edgeTrainer->addFeatureVecs(featureVector1, pGt1[x], featureVector2, pGt1[x-1]);
			edgeTrainer->addFeatureVecs(featureVector2, pGt1[x-1], featureVector1, pGt1[x]);

			for (word f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pFv2[nFeatures * x + f];		// featureVector2 = fv[x][y-1]
			edgeTrainer->addFeatureVecs(featureVector1, pGt1[x], featureVector2, pGt2[x]);
			edgeTrainer->addFeatureVecs(featureVector2, pGt2[x], featureVector1, pGt1[x]);
		} // x
	} // y
	edgeTrainer->train();

	ticks = getTickCount() - ticks;
	printf("Done! (%fms)\n", ticks * 1000 / getTickFrequency());

	// ==================== STAGE 3: Filling the Graph =====================
	printf("Filling the Graph... ");
	ticks = getTickCount();
	Mat nodePot, edgePot;
	for (int y = 0, idx = 0; y < height; y++) {
		byte *pFv1 = fv.ptr<byte>(y);
		byte *pFv2 = (y > 0) ? fv.ptr<byte>(y - 1) : NULL;
		for (int x = 0; x < width; x++, idx++) {
			for (word f = 0; f < nFeatures; f++) featureVector1.at<byte>(f, 0) = pFv1[nFeatures * x + f];			// featureVector1 = fv[x][y]
			nodePot = nodeTrainer->getNodePotentials(featureVector1);												// node potential
			graph->setNode(idx, nodePot);

			if (x > 0) {
				for (word f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pFv1[nFeatures * (x - 1) + f];	// featureVector2 = fv[x-1][y]
				edgePot = edgeTrainer->getEdgePotentials(featureVector1, featureVector2, params, params_len);		// edge potential
				graph->setArc(idx, idx - 1, edgePot);
			} // if x
			if (y > 0) {
				for (word f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pFv2[nFeatures * x + f];		// featureVector2 = fv[x][y-1]
				edgePot = edgeTrainer->getEdgePotentials(featureVector1, featureVector2, params, params_len);		// edge potential
				graph->setArc(idx, idx - width, edgePot);
			} // if y
		} // x
	} // y
	ticks = getTickCount() - ticks;
	printf("Done! (%fms)\n", ticks * 1000 / getTickFrequency());

	// ========================= STAGE 4: Decoding =========================
	printf("Decoding... ");
	ticks = getTickCount();
	std::vector<byte> optimalDecoding = decoder->decode(10);
	ticks =  getTickCount() - ticks;
	printf("Done! (%fms)\n", ticks * 1000 / getTickFrequency());

	// ====================== Evaluation =======================	
	Mat solution(imgSize, CV_8UC1, optimalDecoding.data());
	confMat->estimate(gt, solution);																				// compare solution with the groundtruth
	char str[255];
	sprintf(str, "Accuracy = %.2f%%", confMat->getAccuracy());
	printf("%s\n", str);

	// ====================== Visualization =======================
	marker->markClasses(img, solution);
	rectangle(img, Point(width - 160, height- 18), Point(width, height), CV_RGB(0,0,0), -1);
	putText(img, str, Point(width - 155, height - 5), FONT_HERSHEY_SIMPLEX, 0.45, CV_RGB(225, 240, 255), 1, CV_AA);
	imwrite(argc[6], img);

	return 0;
}
@endcode
*/

/**
@page demostereo Demo Stereo

add description here

<table align="center">
<tr>
<td colspan="2"><center><b>Input stereo pair</b></center></td>
<td></td>
<td><center><b>Output</b></center></td>
</tr>
<tr>
<td><img src="tsukuba_left.jpg"></td>
<td><img src="tsukuba_right.jpg"></td>
<td><img src="arrow.png"></td>
<td><img src="tsukuba_resdisp.jpg"></td>
</tr>
<tr>
<td><center><b>tsukuba_left.jpg</b></center></td>
<td><center><b>tsukuba_right.jpg</b></center></td>
<td></td>
<td><center><b>Resulting Disparity Map</b></center></td>
</tr>
</table>

@code
#include "DGM.h"
using namespace DirectGraphicalModels;

int main(int argc, char *argv[])
{
	// Reading parameters and images
	Mat		  imgL			= imread("tsukuba_left.jpg", 0);
	Mat		  imgR			= imread("tsukuba_right.jpg", 0);
	int		  minDisparity	= 5;
	int		  maxDisparity	= 16;
	int		  width			= imgL.cols;
	int		  height		= imgL.rows;
	
	const unsigned int nStates = maxDisparity - minDisparity;

	CGraph	* graph			= new CGraph(nStates);
	CInfer	* decoder		= new CInferTRW(graph);

	Mat nodePot(nStates, 1, CV_32FC1);										// node Potential (column-vector)
	Mat edgePot(nStates, nStates, CV_32FC1);								// edge Potential

	// No training
	// Defynig the edge potential
	edgePot = CTrainEdgePotts::getEdgePotentials(1.175f, nStates);
	// equivalent to:
	// ePot.at<float>(0, 0) = 1.175;	ePot.at<float>(0, 1) = 1;
	// ePot.at<float>(1, 0) = 1;		ePot.at<float>(1, 1) = 1.175;

	// ==================== Building and filling the graph ====================
	for (int y = 0; y < height; y++) {
		byte * pImgL	= imgL.ptr<byte>(y);
		byte * pImgR	= imgR.ptr<byte>(y);
		for (int x = 0; x < width; x++) {
			float imgL_value = static_cast<float>(pImgL[x]);
			for (unsigned int s = 0; s < nStates; s++) {					// states
				int disparity = minDisparity + s;
				float imgR_value = (x + disparity < width) ? static_cast<float>(pImgR[x + disparity]) : imgL_value;
				float p = 1.0f - fabs(imgL_value - imgR_value) / 255.0f;
				nodePot.at<float>(s, 0) = p * p;
			} // s

			size_t idx = graph->addNode(nodePot);
			if (x > 0) graph->addArc(idx, idx - 1, edgePot);
			if (y > 0) graph->addArc(idx, idx - width, edgePot);
		} // x
	} // y

	// =============================== Decoding ===============================
	printf("Decoding... ");
	int64 ticks = getTickCount();
	vec_byte_t optimalDecoding = decoder->decode(100);
	ticks = getTickCount() - ticks;
	printf("Done! (%fms)\n", ticks * 1000 / getTickFrequency());

	// ============================ Visualization =============================
	Mat disparity(imgL.size(), CV_8UC1, optimalDecoding.data());
	disparity = (disparity + minDisparity) * (256 / maxDisparity);
	medianBlur(disparity, disparity, 3);

	imshow("Disparity", disparity);

	cvWaitKey();

	return 0;
}
@endcode
*/