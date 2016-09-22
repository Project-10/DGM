#pragma once

#include "VIS\Marker.h"
#include "VIS\MarkerGraph.h"
#include "VIS\MarkerHistogram.h"

/**
@defgroup moduleVIS VIS Module
@section sec_marker_title Visualization Module

Marker module provides with a variety of tools for visualizing and analyzing used data as well as intermediate and final results:
- <b>markClasses</b>         : DirectGraphicalModels::vis::CMarker::markClasses()            for visualizing the classification results;
- <b>drawPotentials</b>      : DirectGraphicalModels::vis::CMarker::drawPotentials()         for visualizing the node and edge potentials;
- <b>drawConfusionMatrix</b> : DirectGraphicalModels::vis::CMarker::drawConfusionMatrix()    for visualizing and analyzing the qualitative classification results;
- <b>drawHistogram</b>       : DirectGraphicalModels::vis::CMarkerHistogram::drawHistogram() for visualizing and analyzing distribution of features for given classes;
- <b>drawDictionary</b>      : DirectGraphicalModels::vis::drawDictionary()                  for visualizing the sparse dictionaries;
- <b>drawGraph</b>           : DirectGraphicalModels::vis::drawGraph()                       for visualizing the graphical models;

For user interaction capacity, there are more functions, which allow for handling the mouse clicks over the figures. Please see our tutorial @ref demovis for more details.

@author Sergey G. Kosov, sergey.kosov@project-10.de

*/

/**
@page demovis Demo Visualization
DGM library has a rich set of tools for visualizing the data, intermediate and final results, as well as for interaction with created figures. It also provides  tools for analyzing the classification accuracy.
In this example we took the @ref demotrain tutorial, simplified the training sections and expanded the visulaization part. First we estimate the quality of classification with
<a href="https://en.wikipedia.org/wiki/Confusion_matrix">confusion matrix</a>, then we visualize the feature distributions for defined classes in the training dataset as well as the node and edge potentials.
For user interaction capacity we define additional functions for handling the mouse clicks over the figures.

<table align="center">
<tr>
<td><center><b>Confusion matrix</b></center></td>
<td><center><b>Edge potential matrix</b></center></td>
<td><center><b>Node potential vector</b></center></td>
<td><center><b>Feature distribution histograms</b></center></td>
</tr>
<tr>
<td valign="top"><img src="confusion_matrix.jpg"></td>
<td valign="top"><img src="edge_potential.jpg"></td>
<td valign="top"><img src="node_potential.jpg"></td>
<td valign="top"><img src="histogram.jpg"></td>
</tr>
</table>

@code
#include "DGM.h"
#include "VIS.h"
using namespace DirectGraphicalModels;
using namespace DirectGraphicalModels::vis;

// Global definitions
Mat histogramImg;

typedef struct {
	CGraph				* pGraph;
	CMarkerHistogram	* pMarker;
	int					  imgWidth;
} USER_DATA;

int main(int argv, char *argc[])
{
	const CvSize		imgSize		= cvSize(400, 400);
	const int			width		= imgSize.width;
	const int			height		= imgSize.height;
	const unsigned int	nStates		= 6;		// { road, traffic island, grass, agriculture, tree, car }
	const unsigned int	nFeatures	= 3;

	if (argv != 4) {
		print_help();
		return 0;
	}

	// Reading parameters and images
	Mat img			= imread(argc[1], 1); resize(img, img, imgSize, 0, 0, INTER_LANCZOS4);		// image
	Mat fv			= imread(argc[2], 1); resize(fv,  fv,  imgSize, 0, 0, INTER_LANCZOS4);		// feature vector
	Mat gt			= imread(argc[3], 0); resize(gt,  gt,  imgSize, 0, 0, INTER_NEAREST);		// groundtruth

	CTrainNode			* nodeTrainer	 = new CTrainNodeNaiveBayes(nStates, nFeatures);
	CTrainEdge			* edgeTrainer	 = new CTrainEdgePottsCS(nStates, nFeatures);
	float				  params[]		 = { 400, 0.001f };
	size_t				  params_len	 = 2;
	CGraph				* graph			 = new CGraph(nStates);
	CInfer				* decoder		 = new CInferLBP(graph);
	// Define custom colors in RGB format for our classes (for visualization)
	vec_nColor_t		  palette;
	palette.push_back(std::make_pair(CV_RGB(64,  64,   64), "road"));
	palette.push_back(std::make_pair(CV_RGB(0,  255,  255), "tr. island"));
	palette.push_back(std::make_pair(CV_RGB(0,   255,   0), "grass"));
	palette.push_back(std::make_pair(CV_RGB(200, 135,  70), "agricult."));
	palette.push_back(std::make_pair(CV_RGB(64,  128,   0), "tree"));
	palette.push_back(std::make_pair(CV_RGB(255,   0,   0), "car"));
	// Define feature names for visualization
	vec_string_t		  featureNames	= {"NDVI", "Var. Int.", "Saturation"};
	CMarkerHistogram	* marker		= new CMarkerHistogram(nodeTrainer, palette, featureNames);
	CCMat				* confMat		= new CCMat(nStates);

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
	nodeTrainer->addFeatureVec(fv, gt);										// Only Node Training
	nodeTrainer->train();													// Contrast-Sensitive Edge Model requires no training
	ticks = getTickCount() - ticks;
	printf("Done! (%fms)\n", ticks * 1000 / getTickFrequency());

	// ==================== STAGE 3: Filling the Graph =====================
	printf("Filling the Graph... ");
	ticks = getTickCount();
	Mat featureVector1(nFeatures, 1, CV_8UC1);
	Mat featureVector2(nFeatures, 1, CV_8UC1);
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
	imshow("Solution", img);

	// Feature distribution histograms
	histogramImg = marker->drawHistogram();
	imshow("Histogram", histogramImg);

	// Confusion matrix
	Mat cMat	= confMat->getConfusionMatrix();
	Mat cMatImg	= marker->drawConfusionMatrix(cMat, MARK_BW);
	imshow("Confusion Matrix", cMatImg);

	// Setting up handlers
	USER_DATA userData;
	userData.pGraph		= graph;
	userData.pMarker	= marker;
	userData.imgWidth	= width;
	cvSetMouseCallback("Solution",  solutiontWindowMouseHandler, &userData);
	cvSetMouseCallback("Histogram", histogramWindowMouseHandler, &userData);

	cvWaitKey();

	return 0;
}
@endcode

<b>Mouse Handler for drawing the node and edge potentials</b><br>
This mouse handler provides us with the capacity of user interaction with the \b Solution window. By clicking on the pixel of the solution image, we can derive the node potential vector from the graph node,
associated with the chosen pixel, and visualize it. In this example, each graph node has four edges, which connect the node with its direct four neighbors; we visualize one edge potential matrix,
corresponding to one of these four edge potentials. The visualization of the node and edge potentials helps to analyze the local potential patterns.

@code
void solutiontWindowMouseHandler(int Event, int x, int y, int flags, void *param)
{
	USER_DATA	* pUserData	= static_cast<USER_DATA *>(param);
	if (Event == CV_EVENT_LBUTTONDOWN) {
		Mat			  pot, potImg;
		size_t		  node_id	= pUserData->imgWidth * y + x;

		// Node potential
		pUserData->pGraph->getNode(node_id, pot);
		potImg = pUserData->pMarker->drawPotentials(pot, MARK_BW);
		imshow("Node Potential", potImg);

		// Edge potential
		vec_size_t child_nodes;
		pUserData->pGraph->getChildNodes(node_id, child_nodes);
		if (child_nodes.size() > 0) {
			pUserData->pGraph->getEdge(node_id, child_nodes.at(0), pot);
			potImg = pUserData->pMarker->drawPotentials(pot, MARK_BW);
			imshow("Edge Potential", potImg);
		}

		pot.release();
		potImg.release();
	}
}
@endcode

<b>Mouse Handler for drawing the feature distribution histograms</b><br>
This mouse handler allows for user interaction with \b Histogram window. Its capable to visualize the feature distributions separately for each class. User can chose the needed class by clicking on the
color box near to the class name. These feature distributions allow for analyzing the separability of the classes in the feature sapce.

@code
void histogramWindowMouseHandler(int Event, int x, int y, int flags, void *param)
{
	USER_DATA	* pUserData	= static_cast<USER_DATA *>(param);
	if (Event == CV_EVENT_LBUTTONDOWN) {
		CvScalar color;
		color.val[0] = histogramImg.at<byte>(y, 3 * x + 0);	// Blue
		color.val[1] = histogramImg.at<byte>(y, 3 * x + 1);	// Green
		color.val[2] = histogramImg.at<byte>(y, 3 * x + 2);	// Red

		histogramImg.release();
		histogramImg = pUserData->pMarker->drawHistogram(color);
		imshow("Histogram", histogramImg);
	}
}
@endcode


*/