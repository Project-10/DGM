// Example "GraphCuts" 2D-case with LBP decoding
// (http://www.cs.ubc.ca/~schmidtm/Software/UGM/graphCuts.html)
#include "DGM.h"
#include "DGM/timer.h"

using namespace DirectGraphicalModels;

void print_help(char *argv0)
{
	printf("Usage: %s original_image noised_image\n", argv0);
}

int main(int argc, char *argv[]) 
{
	if (argc != 3) {
		print_help(argv[0]);
		return 0;
	}

	const unsigned int	nStates = 2;			// { true; false }

	// Reading parameters and images
	Mat		  img		= imread(argv[1], 0);	if (img.empty())   printf("Can't open %s\n", argv[1]);
	Mat		  noise		= imread(argv[2], 0);	if (noise.empty()) printf("Can't open %s\n", argv[2]);
	
	CGraphPairwise		graph(nStates);
	CGraphPairwiseExt	graphExt(graph, GRAPH_EDGES_GRID | GRAPH_EDGES_DIAG);
	CInferViterbi		decoder(graph);
	
	// no training
	vec_mat_t p(nStates);
	noise.convertTo(p[0], CV_32FC1, -1.0 / 255, 1.0);	// p_true  = 1 - noise / 255
	noise.convertTo(p[1], CV_32FC1, 1.0 / 255);			// p_false = noise / 255
	Mat nodePot;
	merge(p, nodePot);

	// ==================== Building and filling the graph ====================
	graphExt.setGraph(nodePot);
	graphExt.addDefaultEdgesModel(10000, 3);

	// =============================== Decoding ===============================
	Timer::start("Decoding... ");
	vec_byte_t optimalDecoding = decoder.decode(100);
	Timer::stop();
	
	// ====================== Evaluation / Visualization ======================
	noise = Mat(noise.size(), CV_8UC1, optimalDecoding.data()) * 255;
	medianBlur(noise, noise, 3);

	float error = 0;
	for (int y = 0; y < img.rows; y++)
		for (int x = 0; x < img.cols; x++)
			if (noise.at<byte>(y,x) != img.at<byte>(y,x)) error++;

	printf("Accuracy  = %.2f%%\n", 100 - 100 * error / (img.cols * img.rows));
	
	imshow("image", noise);	

	cv::waitKey();

	return 0;
}
