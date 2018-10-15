// Example Disparity estimation with CRFs
#include "DGM.h"
#include "DGM/timer.h"

using namespace DirectGraphicalModels;

void print_help(char *argv0)
{
	printf("Usage: %s left_image right_image min_disparity max_disparity\n", argv0);
}

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

	CGraphPairwise graph(nStates);
	CInferTRW decoder(graph);

	Mat nodePot(nStates, 1, CV_32FC1);										// node Potential (column-vector)
	Mat edgePot(nStates, nStates, CV_32FC1);								// edge Potential	

	// No training
	// Defynig the edge potential
	edgePot = CTrainEdge::getDefaultEdgePotentials(1.175f, nStates);
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

			size_t idx = graph.addNode(nodePot);
			if (x > 0) graph.addArc(idx, idx - 1, edgePot);
			if (y > 0) graph.addArc(idx, idx - width, edgePot);
		} // x
	} // y

	// =============================== Decoding ===============================
	Timer::start("Decoding... ");
	vec_byte_t optimalDecoding = decoder.decode(100);
	Timer::stop();
	
	// ============================ Visualization =============================
	Mat disparity(imgL.size(), CV_8UC1, optimalDecoding.data());
	disparity = (disparity + minDisparity) * (256 / maxDisparity);
	medianBlur(disparity, disparity, 3);

	imshow("Disparity", disparity);

	cv::waitKey();

	return 0;
}
