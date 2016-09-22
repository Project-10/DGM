// Example Disparity estimation with CRFs
#include "DGM.h"
using namespace DirectGraphicalModels;

void print_help(void)
{
	printf("Usage: \"Demo Stereo.exe\" left_image right_image true_disparity\n");
}

int main(int argc, char *argv[])
{
	const unsigned int maxDisparity = 16;
	const unsigned int minDisparity = 4;
	const unsigned int nStates = maxDisparity - minDisparity;

	
	if (argc != 4) {
		print_help();
		return 0;
	}
	
	// Reading parameters and images
	Mat		  imgL		= imread(argv[1], 0);
	Mat		  imgR		= imread(argv[2], 0);
	Mat		  gt		= imread(argv[3], 0);
	// TODO:assert equal sizes
	int		  width		= imgL.cols;
	int		  height	= imgL.rows;

	CGraph	* graph		= new CGraph(nStates);
	CInfer	* decoder	= new CInferViterbi(graph);
	
	Mat nodePot(nStates, 1, CV_32FC1);										// node Potential (column-vector)
	Mat edgePot(nStates, nStates, CV_32FC1);								// edge Potential	

	float params[] = { 100.0f, 1.0f };
	CTrainEdge * edgeTrainer = new CTrainEdgePottsCS(nStates, 1, eP_APP_PEN_CHAR);

	// No training
	// Defynig the edge potential
	edgePot = CTrainEdgePotts::getEdgePotentials(1.075f, nStates);

	// ==================== Building and filling the graph ====================
	for (int y = 0; y < height; y++) {
		byte * pImgL	= imgL.ptr<byte>(y);
		byte * pImgLy	= y > 0 ? imgL.ptr<byte>(y - 1) : NULL;
		byte * pImgR	= imgR.ptr<byte>(y);
		for (int x = 0; x < width; x++) {
			float imgL_value = static_cast<float>(pImgL[x]);
			for (unsigned int d = minDisparity; d < maxDisparity; d++) {					// disparity
				float imgR_value = (x + d < width) ? static_cast<float>(pImgR[x + d]) : imgL_value;
				float delta_value = fabs(imgL_value - imgR_value) / 255;		// \in [0; 1]
				float p = 100 * (1.0f - delta_value);
				nodePot.at<float>(d - minDisparity, 0) = p * p;
			}

			size_t idx = graph->addNode(nodePot);
			if (x > 0)						graph->addArc(idx, idx - 1,			edgeTrainer->getEdgePotentials(Mat(1, 1, CV_8UC1, cvScalar(pImgL[x])), Mat(1, 1, CV_8UC1, cvScalar(pImgL [x - 1])), params, 2, 0.01f));
			if (y > 0)						graph->addArc(idx, idx - width,		edgeTrainer->getEdgePotentials(Mat(1, 1, CV_8UC1, cvScalar(pImgL[x])), Mat(1, 1, CV_8UC1, cvScalar(pImgLy[x])),     params, 2, 0.01f));
			if ((x > 0) && (y > 0))			graph->addArc(idx, idx - width - 1, edgeTrainer->getEdgePotentials(Mat(1, 1, CV_8UC1, cvScalar(pImgL[x])), Mat(1, 1, CV_8UC1, cvScalar(pImgLy[x - 1])), params, 2, 0.01f));
			if ((x < width - 1) && (y > 0)) graph->addArc(idx, idx - width + 1, edgeTrainer->getEdgePotentials(Mat(1, 1, CV_8UC1, cvScalar(pImgL[x])), Mat(1, 1, CV_8UC1, cvScalar(pImgLy[x + 1])), params, 2, 0.01f));
		} // x
	} //

	// =============================== Decoding ===============================
	printf("Decoding... ");
	int64 ticks = getTickCount();
	vec_byte_t optimalDecoding = decoder->decode(100);
	ticks = getTickCount() - ticks;
	printf("Done! (%fms)\n", ticks * 1000 / getTickFrequency());
	
	Mat disparity(imgL.size(), CV_8UC1, optimalDecoding.data());
	disparity = (disparity + minDisparity) * 16;

	float error = 0;
	int   sum = 0;
	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++) 
			if (gt.at<byte>(y, x) > 0) {
				if (disparity.at<byte>(y, x) != gt.at<byte>(y, x)) {
					error++;
					// disparity.at<byte>(y, x) = 255;
				}
				sum++;
			}


	printf("Accuracy  = %.2f%%\n", 100 - 100 * error / sum);

	imshow("disparity", disparity);

	cvWaitKey();

	return 0;
}