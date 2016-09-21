// Example Disparity estimation with CRFs
#include "DGM.h"
using namespace DirectGraphicalModels;

void print_help(void)
{
	printf("Usage: \"Demo Stereo.exe\" left_image right_image true_disparity\n");
}

int main(int argc, char *argv[])
{
	const int maxDisparity = 16;
	
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

	CGraph	* graph		= new CGraph(maxDisparity);
	CInfer	* decoder	= new CInferViterbi(graph);
	
	Mat nodePot(maxDisparity, 1, CV_32FC1);										// node Potential (column-vector)
	Mat edgePot(maxDisparity, maxDisparity, CV_32FC1);							// edge Potential	

	// No training
	// Defynig the edge potential
	edgePot = CTrainEdgePotts::getEdgePotentials(1.075f, maxDisparity);

	// ==================== Building and filling the graph ====================
	for (int y = 0; y < height; y++) {
		byte * pImgL = imgL.ptr<byte>(y);
		byte * pImgR = imgR.ptr<byte>(y);
		for (int x = 0; x < width; x++) {
			float imgL_value = static_cast<float>(pImgL[x]);
			for (int d = 0; d < maxDisparity; d++) {					// disparity
				float imgR_value = (x + d < width) ? static_cast<float>(pImgR[x + d]) : imgL_value;
				float delta_value = fabs(imgL_value - imgR_value) / 255;		// \in [0; 1]
				float p = 100 * (1.0f - delta_value);
				nodePot.at<float>(d, 0) = (d > 4) ? p*p : 0;
			}

			size_t idx = graph->addNode(nodePot);
			if (x > 0) graph->addArc(idx, idx - 1, edgePot);
			if (y > 0) graph->addArc(idx, idx - width, edgePot);
			if ((x > 0) && (y > 0)) graph->addArc(idx, idx - width - 1, edgePot);
			if ((x < width - 1) && (y > 0)) graph->addArc(idx, idx - width + 1, edgePot);
		} // x
	} //

	// =============================== Decoding ===============================
	printf("Decoding... ");
	int64 ticks = getTickCount();
	vec_byte_t optimalDecoding = decoder->decode(100);
	ticks = getTickCount() - ticks;
	printf("Done! (%fms)\n", ticks * 1000 / getTickFrequency());
	
	Mat disparity(imgL.size(), CV_8UC1, optimalDecoding.data());
	disparity *= 16;

	imshow("disparity", disparity);

	cvWaitKey();

	return 0;
}