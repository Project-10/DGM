// Example "GraphCuts" 2D-case with LBP decoding
// (http://www.di.ens.fr/~mschmidt/Software/UGM/graphCuts.html)
#include "DGM.h"
using namespace DirectGraphicalModels;

void print_help(void)
{
	printf("Usage: \"Demo 2D.exe\" original_image noised_image\n");
}

int main(int argc, char *argv[]) 
{
	const unsigned int	nStates	= 2;			// {true; false}
	
	if (argc != 3) {
		print_help();
		return 0;
	}
	
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
	edgePot = CTrainEdgePotts::getEdgePotentials(10000, 2);
	// equivalent to:
	// ePot.at<float>(0, 0) = 1000;	ePot.at<float>(0, 1) = 1;
	// ePot.at<float>(1, 0) = 1;	ePot.at<float>(1, 1) = 1000;

	// ==================== Building and filling the graph ====================
	for (int x = 0; x < width; x++)
		for (int y = 0; y < height; y++) {
			float p = 1.0f - static_cast<float>(noise.at<byte>(y,x)) / 255.0f;
			nodePot.at<float>(0, 0) = p;
			nodePot.at<float>(1, 0) = 1.0f - p;
			size_t idx = graph->addNode(nodePot);
			if (y > 0) graph->addArk(idx, idx - 1, edgePot);
			if (x > 0) graph->addArk(idx, idx - width, edgePot);
			if ((y > 0) && (x > 0)) graph->addArk(idx, idx - width - 1, edgePot);
		} // y

	// =============================== Decoding ===============================
	printf("Decoding... ");
	int64 ticks = getTickCount();
	byte *optimalDecoding = decoder->decode(100);
	ticks =  getTickCount() - ticks;
	printf("Done! (%fms)\n", ticks * 1000 / getTickFrequency());

	
	// ====================== Evaluation / Visualization ======================
	for (int x = 0, i = 0; x < width; x++)
		for (int y = 0; y < height; y++)
			noise.at<byte>(y,x) = 255 * optimalDecoding[i++];
	medianBlur(noise, noise, 3);

	float error = 0;
	for (int x = 0; x < width; x++)
		for (int y = 0; y < height; y++)
			if (noise.at<byte>(y,x) != img.at<byte>(y,x)) error++;

	printf("Accuracy  = %.2f%%\n", 100 - 100 * error / (width * height));
	
	cvNamedWindow("image", CV_WINDOW_AUTOSIZE);
	imshow("image", noise);	
	imwrite("D:\\aaa.png", noise);
	cvWaitKey();

	return 0;
}