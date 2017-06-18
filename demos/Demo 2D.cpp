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

	const unsigned int	nStates = 2;			// {true; false}

	// Reading parameters and images
	Mat		  img		= imread(argv[1], 0);
	Mat		  noise		= imread(argv[2], 0);
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