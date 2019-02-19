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
	Mat		  imgL			= imread(argv[1], 0);	if (imgL.empty()) printf("Can't open %s\n", argv[1]);
	Mat		  imgR			= imread(argv[2], 0);	if (imgR.empty()) printf("Can't open %s\n", argv[2]);
	int		  minDisparity	= atoi(argv[3]);
	int		  maxDisparity	= atoi(argv[4]);
	int		  width			= imgL.cols;
	int		  height		= imgL.rows;
	unsigned int nStates	= maxDisparity - minDisparity;

	CGraphPairwiseKit graphKit(nStates, INFER::TRW);

	// No training
	graphKit.getGraphExt().buildGraph(imgL.size());
	graphKit.getGraphExt().addDefaultEdgesModel(1.175f);

	// ==================== Building and filling the graph ====================
	Mat nodePot(nStates, 1, CV_32FC1);										// node Potential (column-vector)
	size_t idx = 0;
	for (int y = 0; y < height; y++) {
		byte * pImgL	= imgL.ptr<byte>(y);
		byte * pImgR	= imgR.ptr<byte>(y);
		for (int x = 0; x < width; x++) {
			float imgL_value = static_cast<float>(pImgL[x]);
			for (unsigned int s = 0; s < nStates; s++) {					// state
				int disparity = minDisparity + s;
				float imgR_value = (x + disparity < width) ? static_cast<float>(pImgR[x + disparity]) : imgL_value;
				float p = 1.0f - fabs(imgL_value - imgR_value) / 255.0f;
				nodePot.at<float>(s, 0) = p * p;
			}

			graphKit.getGraph().setNode(idx++, nodePot);
		} // x
	} // y

	// =============================== Decoding ===============================
	Timer::start("Decoding... ");
	vec_byte_t optimalDecoding = graphKit.getInfer().decode(0);
	Timer::stop();
	
	// ============================ Visualization =============================
	Mat disparity(imgL.size(), CV_8UC1, optimalDecoding.data());
	disparity = (disparity + minDisparity) * (256 / maxDisparity);
	medianBlur(disparity, disparity, 3);

	imshow("Disparity", disparity);

	waitKey();

	return 0;
}
