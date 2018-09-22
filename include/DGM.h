#pragma once

#include "DGM/CMat.h"
#include "DGM/AveragePrecision.h"
#include "DGM/FeaturesConcatenator.h"
#include "DGM/KDGauss.h"
#include "DGM/KDTree.h"
#include "DGM/random.h"
#include "DGM/parallel.h"

#include "DGM/IPDF.h"
#include "DGM/PDFHistogram.h"
#include "DGM/PDFHistogram2D.h"
#include "DGM/PDFGaussian.h"

#include "DGM/Prior.h"
#include "DGM/PriorNode.h"
#include "DGM/PriorEdge.h"
#include "DGM/PriorTriplet.h"

#include "DGM/ITrain.h"
#include "DGM/TrainNode.h"
#include "DGM/TrainNodeCvKNN.h"
#include "DGM/TrainNodeKNN.h"
#include "DGM/TrainNodeNaiveBayes.h"
#include "DGM/TrainNodeGM.h"
#include "DGM/TrainNodeGMM.h"
#include "DGM/TrainNodeCvGMM.h"
#include "DGM/TrainNodeCvGM.h"
#include "DGM/TrainNodeCvSVM.h"
#include "DGM/TrainNodeCvRF.h"
#include "DGM/TrainNodeMsRF.h"
#include "DGM/TrainNodeCvANN.h"
#include "DGM/TrainEdge.h"
#include "DGM/TrainEdgePotts.h"
#include "DGM/TrainEdgePottsCS.h"
#include "DGM/TrainEdgePrior.h"
#include "DGM/TrainEdgeConcat.h"
#include "DGM/TrainTriplet.h"
#include "DGM/TrainLink.h"
#include "DGM/TrainLinkNested.h"

#include "DGM/GraphDense.h"
#include "DGM/GraphDenseExt.h"
#include "DGM/IGraphPairwise.h"
#include "DGM/GraphPairwise.h"
#include "DGM/GraphWeiss.h"
#include "DGM/GraphPairwiseExt.h"
#include "DGM/Graph3.h"

#include "DGM/Infer.h"
#include "DGM/InferExact.h"
#include "DGM/InferDense.h"
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
	- @ref moduleLFEX for pixel-level features.
	- @ref moduleGFEX for image-level features.
- Visualization @ref moduleVIS, wich allows for visualizing the used data and features as well as intermediate and final classification results.

These tasks are optimized for speed, @a i.e. high-efficient calculations. The code is written entirely in C++ and can be compiled with Microsoft Visual C++.

@section sec_main_overview Methods Overview
@subsection sec_main_train Training
DGM implements the following training methods: 

@subsubsection sec_main_train_nodes Nodes | Unary Potentials
- <b>NaiveBayes:</b> Naive Bayes training @ref DirectGraphicalModels::CTrainNodeBayes
- <b>GMM:</b> Gaussian Mixture Model training (<a href="http://www.project-10.de/Kosov/files/GCPR_2013.pdf" target="_blank">Sequential GMM Training Algorithm</a>) @ref DirectGraphicalModels::CTrainNodeGMM
- <b>CvGMM:</b> OpenCV Gaussian Mixture Model training @ref DirectGraphicalModels::CTrainNodeCvGMM
- <b>KNN:</b> <i>k</i>-Nearest Neighbors training @ref DirectGraphicalModels::CTrainNodeKNN
- <b>CvKNN</b> OpenCV <i>k</i>-Nearest Neighbors training @ref DirectGraphicalModels::CTrainNodeCvKNN
- <b>CvRF:</b> OpenCV Random Forest training @ref DirectGraphicalModels::CTrainNodeCvRF
- <b>MsRF:</b> Microsoft Research Random Forest training @ref DirectGraphicalModels::CTrainNodeMsRF
- <b>CvANN:</b> OpenCV Artificial Neural Network training @ref DirectGraphicalModels::CTrainNodeCvANN
- <b>CvSVM:</b> OpenCV Support Vector Machine training @ref DirectGraphicalModels::CTrainNodeCvSVM

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
- <b>Dense:</b> <a href="http://vladlen.info/publications/efficient-inference-in-fully-connected-crfs-with-gaussian-edge-potentials/" target="_blank">Efficient inference for \a dense CRFs with Gaussian edge potentials</a> @ref DirectGraphicalModels::CGraphDense

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
- <b>Gauss:</b> Sampling from the multivariate gaussian distribution @ref DirectGraphicalModels::CKDGauss::getSample()

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
DGM is a cross-platform C++ library. The description here was tested on Windows 10 / Visual Studio 2017 and MacOS High Sierra 10.13.6 / Xcode 9.4.1. If you encounter errors after following the steps described below, 
feel free to contact us via our <a href="http://project-10.de/forum/viewforum.php?f=31">User Q&A forum</a>. We'll do our best to help you out.<br>

DGM has only one dependency: it is based on <a href="https://www.opencv.org/">OpenCV</a> library.
In order to use the DGM library, the OpenCV library should be also installed.

@section sec_install_cv Installing OpenCV
- Download the latest version of the OpenCV library from <a href="https://www.opencv.org/releases.html" target="_blank">https://www.opencv.org/releases.html</a>
- Build and install the library. You may follow the following guidances: 
	- <a href="https://docs.opencv.org/3.4.2/d3/d52/tutorial_windows_install.html" target="_blank">Installation in Windows guide</a> or 
	- <a href="https://docs.opencv.org/3.4.2/d7/d9f/tutorial_linux_install.html" target="_blank">Installation in Linux guide</a>

@section sec_install_dgm Installing DGM
- Download either the latest stable DGM version from <a href="http://research.project-10.de/dgm/#downloads" target="_blank">Project X Research</a> 
  or fork the latest snapshot from our <a href="https://github.com/Project-10/DGM" target="_blank">GitHub repository</a>
- Unpack it to your local folder (for example to disk @b C:\\ for Windows or to @b /Users/username/ for MacOS, so the library path will be @b C:\\DGM\\ or @b /Users/username/DGM/)
@subsection sec_install_dgm_source Building DGM from Source Using CMake
In case you want to build the library (recommended), follow these instructions, otherwise - skip this step and proceed to @ref sec_install_dgm_built. 
This step also assumes that you have downloaded the sources of the DGM library.
- Download and install <a href="https://cmake.org/download" target="_blank">CMake</a> for your operating system
- Run \b cmake-gui.exe in Windows or \b CMake.app in MacOS
- In the <i>“Where is the source code”</i> field choose the DGM source directory: \b DGM<br>
  In the <i>“Where to build the binaries”</i> field choose directory where Visual Studio or Xcode project files will be generated: \a e.g. \b DGM/build
- Press \a Configure button and choose <i>Visual Studio</i> for using 32-bit compiler, <i>Visual Studio Win64</i> for using 64-bit compiler or <i>Xcode</i> as building environment
- Be sure that the \a OpenCV_DIR is pointing to the OpenCV installation directory (\a e.g. \b OpenCV/build/install or \b /usr/local/share/OpenCV), where \b OpenCVConfig.cmake file is located
- (Optioanlly) you can change \a CMAKE_INSTALL_PREFIX to the directory where the DGM binaries will be installed (\a e.g. to \b DGM/build/install)
- Press one more time \a Configure and then \a Generate, so the IDE project files will be generated in the \b DGM/build
- Open the generated projectd by pressing the \a Open \a Project button or directly by opening file \b DGM/build/DGM.sln or \b DGM/build/DGM.xcodeproj 
- Build \b ALL_BUILD and \b INSTALL projects first for \a Debug and then for \a Release configuration. That will copy DGM headers, binaries and demonstration applications to the install folder \b DGM/build/install
- Windows users may copy the OpenCV binaries into the install folder by executing script \b /DGM/build/install/bin/copyOpenCVDLL.bat
- (Optionally) you can copy the install folder with the ready-to-use DGM library (\a e.g. \b DGM/build/install) to any other folder

@subsection sec_install_dgm_built Using the Pre-built Libraries
This step assumes that you have downloaded DGM-package with the pre-build binaries. In such case the type and version of the downloaded binaries should correspond to your C++ compiler. 
If it is not, please return to the @ref sec_install_dgm_source section and generate the binaries with your compiler. The content of the install folder (\a e.g. \b DGM/build/install) will 
correspond to the downloaded pre-build DGM package. 

@subsection sec_install_dgm_after After Installation
As soosn as the DGM library is installed, you can launch the demo applications from the \b /bin folder. If you have built the binaries from the sources, you can also start the demo projects directly from your IDE.
The corresponding description may be found in @ref demo. Do not hesitate to modify these demo projects for your needs or start your own project based on our demo code. 

If you wish to generate a new projects, which will use DGM, or add DGM to an existing project we highly recomend you to use <a href="https://cmake.org" target="_blank">CMake</a> and follow the 
<a href="http://project-10.de/forum/viewtopic.php?f=31&t=1028&sid=09c4a9156520f7cf81bd474ac278ed51" target="_blank">Using DGM library with CMake</a> guidances, where template \b CMakeLists.txt file is provided.<br>
Alternatively, you can specify the following paths and library in your IDE manually:
- Add to Configuration Properties -> C/C++ -> General -> Additional Include Directories the path \b install_folder/include
- Add to Configuration Properties -> Linker -> General -> Additional Library Directories the path \b install_folder/lib for both Release and Debug configurations
- Add to Configuration Properties -> Linker -> Input -> Additional Dependencies the libraries \b dgm160.lib, \b fex160.lib, \b vis160.lib and \b dgm160d.lib, \b fex160d.lib, \b vis160d.lib 
  for Release and Debug configurations accordingly
*/

/**
@page demo Tutorials
## How to use the code
The documentation for DGM consists of a series of demos, showing how to use DGM to perform various tasks. These demos also contain some tutorial material on graphical models.

### Bacis Tutorials
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

### Advanced Tutorials
- @subpage demorandommodel : An advanced tutorial to the unary potentials training.
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

int main(int argc, char *argv[]) 
{
	if (argc != 3) {
		print_help(argv[0]);
		return 0;
	}

	const unsigned int	nStates = 2;			// {true; false}

	// Reading parameters and images
	Mat		  img		= imread(argv[1], 0);
	Mat		  noise		= imread(argv[2], 0);
	int		  width		= img.cols;
	int		  height	= img.rows;
	
	CGraphPairwise	* graph		= new CGraphPairwise(nStates);
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
	Timer::start("Decoding... ");
	vec_byte_t optimalDecoding = decoder->decode(100);
	Timer::stop();

	
	// ====================== Evaluation / Visualization ======================
	noise = Mat(noise.size(), CV_8UC1, optimalDecoding.data()) * 255;
	medianBlur(noise, noise, 3);

	float error = 0;
	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
			if (noise.at<byte>(y,x) != img.at<byte>(y,x)) error++;

	printf("Accuracy  = %.2f%%\n", 100 - 100 * error / (width * height));
	
	imshow("image", noise);	

	cvWaitKey();

	return 0;
}
@endcode
*/

/**
@page demotrain Demo Train
In this demo, we consider the case when the training data is aviable. In this example the trainig data is represented in form of 
manually labelled images. The original images \b Original \b Image.jpg are color-infrared images, and the grounftruth images \b GroundTruth \b Image.jpg
have 6 different classes, namely \a road, \a traffic \a island, \a grass, \a agriculture, \a tree and \a car (instances of which are not represented in image). 

In this example DGM uses 3 features extacted from the \b Original \b Image.jpg (Please refer to the @ref demofex for details). One image is used for training and 
another - for testing. Finally, we evaluate the results by comparing the lebelled image with the groundtruth.

<table align="center">
<tr>
	<td colspan="3"><center><b>Input Training Images</b></center></td>
</tr>
<tr>
  <td><img src="001_img_small.jpg"></td>
  <td><img src="001_fv_small.jpg"></td>
  <td><img src="001_gt_small.jpg"></td>
</tr>
<tr>
  <td><center><b>Original Image.jpg</b></center></td>
  <td><center><b>Feature Vector.jpg</b></center></td>
  <td><center><b>Ground Truth.jpg</b></center></td>
</tr>
</table>

<hr>

<table align="center">
<tr>
<td colspan="3"><center><b>Input Testing Images</b></center></td>
<td></td>
<td><center><b>Output</b></center></td>
</tr>
<tr>
<td><img src="002_img_small.jpg"></td>
<td><img src="002_fv_small.jpg"></td>
<td><img src="002_gt_small.jpg"></td>
<td><img src="arrow.png"></td>
<td><img src="002_res_small.jpg"></td>
</tr>
<tr>
<td><center><b>Original Image.jpg</b></center></td>
<td><center><b>Feature Vector.jpg</b></center></td>
<td><center><b>Ground Truth.jpg</b> (for evaluation)</center></td>
<td></td>
<td><center><b>Resulting Class Map</b></center></td>
</tr>
</table>

@code
#include "DGM.h"
#include "VIS.h"
#include "DGM\timer.h"
using namespace DirectGraphicalModels;
using namespace DirectGraphicalModels::vis;

int main(int argc, char *argv[])
{
	const CvSize		imgSize		= cvSize(400, 400);
	const int			width		= imgSize.width;
	const int			height		= imgSize.height;
	const unsigned int	nStates		= 6;		// {road, traffic island, grass, agriculture, tree, car}
	const unsigned int	nFeatures	= 3;

	if (argc != 9) {
		print_help(argv[0]);
		return 0;
	}

	// Reading parameters and images
	int nodeModel	= atoi(argv[1]);																	// node training model
	int edgeModel	= atoi(argv[2]);																	// edge training model
	Mat train_fv	= imread(argv[3], 1); resize(train_fv, train_fv, imgSize, 0, 0, INTER_LANCZOS4);	// training image feature vector
	Mat train_gt	= imread(argv[4], 0); resize(train_gt, train_gt, imgSize, 0, 0, INTER_NEAREST);		// groundtruth for training
	Mat test_fv		= imread(argv[5], 1); resize(test_fv,  test_fv,  imgSize, 0, 0, INTER_LANCZOS4);	// testing image feature vector
	Mat test_gt		= imread(argv[6], 0); resize(test_gt,  test_gt,  imgSize, 0, 0, INTER_NEAREST);		// groundtruth for evaluation
	Mat test_img	= imread(argv[7], 1); resize(test_img, test_img, imgSize, 0, 0, INTER_LANCZOS4);	// testing image

	CTrainNode		* nodeTrainer	= NULL;
	CTrainEdge		* edgeTrainer	= NULL;
	CGraphExt		* graph			= new CGraphExt(nStates);
	CInfer			* decoder		= new CInferLBP(graph);
	CMarker			* marker		= new CMarker(DEF_PALETTE_6);
	CCMat			* confMat		= new CCMat(nStates);
	float			  params[]		= {100, 0.01f};
	size_t			  params_len;

	switch(nodeModel) {
		case 0: nodeTrainer = new CTrainNodeBayes(nStates, nFeatures);	    break;
		case 1: nodeTrainer = new CTrainNodeGMM(nStates, nFeatures);		break;
		case 2: nodeTrainer = new CTrainNodeCvGMM(nStates, nFeatures);		break;
		case 3: nodeTrainer = new CTrainNodeKNN(nStates, nFeatures);		break;
		case 4: nodeTrainer = new CTrainNodeCvRF(nStates, nFeatures);		break;
#ifdef USE_SHERWOOD
		case 5: nodeTrainer = new CTrainNodeMsRF(nStates, nFeatures);		break;
#endif
		default: printf("Unknown node_training_model is given\n"); print_help(argv[0]); return 0;
	}
	switch(edgeModel) {
		case 0: params[0] = 1;	// Emulate "No edges"
		case 1:	edgeTrainer = new CTrainEdgePotts(nStates, nFeatures);		params_len = 1; break;
		case 2:	edgeTrainer = new CTrainEdgePottsCS(nStates, nFeatures);	params_len = 2; break;
		case 3:	edgeTrainer = new CTrainEdgePrior(nStates, nFeatures);		params_len = 2; break;
		case 4:
			edgeTrainer = new CTrainEdgeConcat<CTrainNodeBayes, CDiffFeaturesConcatenator>(nStates, nFeatures);
			params_len = 1;
			break;
		default: printf("Unknown edge_training_model is given\n"); print_help(argv[0]); return 0;
	}

	// ==================== STAGE 1: Building the graph ====================
	Timer::start("Building the Graph... ");
	graph->build(imgSize);
	Timer::stop();

	// ========================= STAGE 2: Training =========================
	Timer::start("Training... ");
	// Node Training (compact notation)
	nodeTrainer->addFeatureVec(train_fv, train_gt);

	// Edge Training (comprehensive notation)
	Mat featureVector1(nFeatures, 1, CV_8UC1);
	Mat featureVector2(nFeatures, 1, CV_8UC1);
	for (int y = 1; y < height; y++) {
		byte *pFv1 = train_fv.ptr<byte>(y);
		byte *pFv2 = train_fv.ptr<byte>(y - 1);
		byte *pGt1 = train_gt.ptr<byte>(y);
		byte *pGt2 = train_gt.ptr<byte>(y - 1);
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

	nodeTrainer->train();
	edgeTrainer->train();
	Timer::stop();

	// ==================== STAGE 3: Filling the Graph =====================
	Timer::start("Filling the Graph... ");
	Mat nodePotentials = nodeTrainer->getNodePotentials(test_fv);		// Classification: CV_32FC(nStates) <- CV_8UC(nFeatures)
	graph->setNodes(nodePotentials);									// Filling-in the graph nodes
	graph->fillEdges(edgeTrainer, test_fv, params, params_len);			// Filling-in the graph edges with pairwise potentials
	Timer::stop();

	// ========================= STAGE 4: Decoding =========================
	Timer::start("Decoding... ");
	vec_byte_t optimalDecoding = decoder->decode(100);
	Timer::stop();

	// ====================== Evaluation =======================
	Mat solution(imgSize, CV_8UC1, optimalDecoding.data());
	confMat->estimate(test_gt, solution);								// compare solution with the groundtruth
	char str[255];
	sprintf(str, "Accuracy = %.2f%%", confMat->getAccuracy());
	printf("%s\n", str);

	// ====================== Visualization =======================
	marker->markClasses(test_img, solution);
	rectangle(test_img, Point(width - 160, height- 18), Point(width, height), CV_RGB(0,0,0), -1);
	putText(test_img, str, Point(width - 155, height - 5), FONT_HERSHEY_SIMPLEX, 0.45, CV_RGB(225, 240, 255), 1, CV_AA);
	imwrite(argv[8], test_img);

	imshow("Image", test_img);
	cvWaitKey(0 * 1000);

	return 0;
}
@endcode

Please note, that in this tutorial we used the Extended Graph class: @ref DirectGraphicalModels::CGraphPairwiseExt, which is built upon the regular Graph class and 
adds wrappers for common DGM operations on rectangular images. Among other wrappers it has one wrapper for graph building (used in the Stage 1) and other 
wrappers for nodes / edges classification (used in Stage 3).
*/

/**
@page demorandommodel Demo Random Model

This is an advanced tutorial. Be sure to get through @ref demotrain, @ref demofex and read the <a href="http://project-10.de/forum/viewtopic.php?f=31&t=954" target="blank"> Training of a random model</a> article before proceeding with this tutorial. 

In this tutorial we use only 2 features, in order to be able to visualize the distribution of the training samples at 2-diensional canvas. For sake of simplicity we also limit the number of states (classes) till 3 and show them with pure red, green and blue colors. The real sample distribution, known from the training image, is than approximated with generative and discriminative methods. In order to reconstruct these approximations, we classify a square of 256 x 256 pixels, using one of the unary (node) trainers. Thus the resulting potential map shows us how concrete classifier sees the feature distribution.

Please note, that we use the values of the partition functions, stored in variable Z. Thus, the generative classifiers try to reconstruct the training samples distribution. They do that with the number of inner parameters, which is much less than the number of training samples. The discriminative methods do not aim to reconstruct the training distribution itself, but provide high and normalized potentials for every pixel of the classification area.


@code
#include "DGM.h"
#include "VIS.h"
#include "FEX.h"
#include "DGM\timer.h"
using namespace DirectGraphicalModels;
using namespace DirectGraphicalModels::vis;
using namespace DirectGraphicalModels::fex;

int main(int argc, char *argv[])
{
	const CvSize		imgSize		= cvSize(400, 400);
	const unsigned int	nStates		= 3;	 		
	const unsigned int	nFeatures	= 2;		// {ndvi, saturation}

	if (argc != 5) {
		print_help(argv[0]);
		return 0;
	}
	
	// Reading parameters and images
	int nodeModel	= atoi(argv[1]);
	Mat img			= imread(argv[2], 1); resize(img, img, imgSize, 0, 0, INTER_LANCZOS4);	// training image 
	Mat gt			= imread(argv[3], 0); resize(gt, gt, imgSize, 0, 0, INTER_NEAREST);	    // groundtruth for training
	gt				= shrinkStateImage(gt, nStates);										// reduce the number of classes in gt to nStates

	float Z;																				// the value of partition function
	CTrainNode	* nodeTrainer = NULL;
	switch(nodeModel) {
		case 0: nodeTrainer = new CTrainNodeBayes(nStates, nFeatures);	    Z = 2e34f; break;
		case 1: nodeTrainer = new CTrainNodeGMM(nStates, nFeatures);		Z = 1.0f; break;
		case 2: nodeTrainer = new CTrainNodeCvGMM(nStates, nFeatures);		Z = 1.0f; break;
		case 3: nodeTrainer = new CTrainNodeKNN(nStates, nFeatures);		Z = 1.0f; break;
		case 4: nodeTrainer = new CTrainNodeCvRF(nStates, nFeatures);		Z = 1.0f; break;
#ifdef USE_SHERWOOD
		case 5: nodeTrainer = new CTrainNodeMsRF(nStates, nFeatures);		Z = 1.0f; break;
#endif
		default: printf("Unknown node_training_model is given\n"); print_help(argv[0]); return 0;
	}
	CMarkerHistogram marker(nodeTrainer, DEF_PALETTE_3);

	//	---------- Features Extraction ----------
	vec_mat_t featureVector;
	fex::CCommonFeatureExtractor fExtractor(img);
	featureVector.push_back(fExtractor.getNDVI(0).autoContrast().get());
	featureVector.push_back(fExtractor.getSaturation().invert().get());

	//	---------- Training ----------
	Timer::start("Training... ");
	nodeTrainer->addFeatureVec(featureVector, gt);
	nodeTrainer->train();
	Timer::stop();

	//	---------- Visualization ----------
	if (nodeModel == 0) {
		imshow("histogram 1d", marker.drawHistogram());
		imshow("histogram 2d", marker.drawHistogram2D());
	}

	Timer::start("Classifying...");
	Mat classMap = marker.drawClassificationMap2D(Z);
	Timer::stop();
	imwrite(argv[4], classMap);

	imshow("class map 2d", classMap);
	cvWaitKey(1000);

	return 0;
}
@endcode

The function, which reduces the amount of classes in the training data by merging some classes into one.
@code
Mat shrinkStateImage(const Mat &gt, byte nStates)
{
	Mat res;
	gt.copyTo(res);

	for (auto it = res.begin<byte>(); it != res.end<byte>(); it++)
		if (*it < 3) *it = 0;
		else if (*it < 4) *it = 1;
		else *it = 2;

	return res;
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
	if (argc != 5) {
		print_help(argv[0]);
		return 0;
	}

	// Reading parameters and images
	Mat		  imgL			= imread(argv[1], 0);
	Mat		  imgR			= imread(argv[2], 0);
	int		  minDisparity	= atoi(argv[3]);
	int		  maxDisparity	= atoi(argv[4]);
	int		  width			= imgL.cols;
	int		  height		= imgL.rows;
	unsigned int nStates	= maxDisparity - minDisparity;

	CGraphPairwise	* graph		= new CGraphPairwise(nStates);
	CInfer	* decoder	= new CInferTRW(graph);

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
			for (unsigned int s = 0; s < nStates; s++) {						// state
				int disparity = minDisparity + s;
				float imgR_value = (x + disparity < width) ? static_cast<float>(pImgR[x + disparity]) : imgL_value;
				float p = 1.0f - fabs(imgL_value - imgR_value) / 255.0f;
				nodePot.at<float>(s, 0) = p * p;
			}

			size_t idx = graph->addNode(nodePot);
			if (x > 0) graph->addArc(idx, idx - 1, edgePot);
			if (y > 0) graph->addArc(idx, idx - width, edgePot);
		} // x
	} // y

	// =============================== Decoding ===============================
	Timer::start("Decoding... ");
	vec_byte_t optimalDecoding = decoder->decode(100);
	Timer::stop();
	
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
