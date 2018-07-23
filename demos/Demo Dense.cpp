// Example "Dense-CRF" 2D-case with model training
#include "DGM.h"
#include "VIS.h"
#include "DGM\serialize.h"
#include "..\3rdparty\densecrf\densecrf.h"

using namespace DirectGraphicalModels;
using namespace DirectGraphicalModels::vis;

// Store the colors we read, so that we can write them again.
std::vector<Vec3b> vPalette;

// Produce a color image from a bunch of labels
template <typename T>
Mat colorize(const Mat &map)
{
	Mat res(map.size(), CV_8UC3);

	for (int y = 0; y < res.rows; y++) {
		const T *pMap = map.ptr<T>(y);
		Vec3b	*pRes = res.ptr<Vec3b>(y);
		for (int x = 0; x < res.cols; x++) 
			pRes[x]= vPalette[pMap[x]];
	}

	return res;
}

// Simple classifier that is 50% certain that the annotation is correct
Mat classify(const Mat &gt, int nStates)
{
	// Certainty that the groundtruth is correct
	const float GT_PROB = 0.55f;

	Mat res(gt.size(), CV_32FC(nStates));
	res.setTo( (1.0f - GT_PROB) / (nStates - 1));
	
	for (int y = 0; y < res.rows; y++) {
		const byte	* pGt  = gt.ptr<byte>(y);
		float		* pRes = res.ptr<float>(y);
		for (int x = 0; x < res.cols; x++) {

			int state = pGt[x];

			//for (int s = 0; s < nStates; s++) pRes[x * nStates + s] = (1.0f - GT_PROB) / (nStates - 1);
			pRes[x * nStates + state] = GT_PROB;
		} // x
	} // y

	return res;
}

void fillPalette(void)
{
	if (!vPalette.empty()) vPalette.clear();
	vPalette.push_back(Vec3b(   0, 0,   255));		
	vPalette.push_back(Vec3b(   0, 128, 255));		
	vPalette.push_back(Vec3b(   0, 255, 255));		
	vPalette.push_back(Vec3b(   0, 255, 128));		
	vPalette.push_back(Vec3b(   0, 255, 0 ));		
	vPalette.push_back(Vec3b( 128, 255, 0 ));		
	vPalette.push_back(Vec3b( 255, 255, 0 ));		
	vPalette.push_back(Vec3b( 255, 128, 0 ));		
	vPalette.push_back(Vec3b( 255, 0,   0 ));		
	vPalette.push_back(Vec3b( 255, 0,   128));		
	vPalette.push_back(Vec3b( 255, 0,   255));		
	vPalette.push_back(Vec3b( 128, 0,   255));		
	vPalette.push_back(Vec3b(   0, 0,   128));		
	vPalette.push_back(Vec3b(   0, 64,  128));		
	vPalette.push_back(Vec3b(   0, 128, 128));		
	vPalette.push_back(Vec3b(   0, 128, 64));		
	vPalette.push_back(Vec3b(   0, 128, 0 ));		
//	vPalette.push_back(Vec3b(  64, 128, 0 ));					
	vPalette.push_back(Vec3b( 128, 128, 0 ));		
	vPalette.push_back(Vec3b( 128,  64, 0 ));		
	vPalette.push_back(Vec3b( 128,   0, 0 ));		
	vPalette.push_back(Vec3b( 128,   0, 64));		
	
}

int main(int argc, char *argv[])
{
	const int nStates = 21;
	
	CCMat	confMat(nStates), confMatGlobal(nStates);

	char path[256];

	fillPalette();

	for (word m = 1; m <= 21; m++) {
		if (m == 17) continue;
		for (word i = 11; i <= 20; i++) {
			if (m == 6 && i == 17) continue;
			if (m == 18 && i == 9) continue;
			printf("#%d-%d: ", m, i);
		
			sprintf(path, "Z:\\Data\\EMDS4\\_Kimiaki\\Original_EM_Images\\t4-g%02d-%02d.png", m < 17 ? m : m - 1, i);
			Mat img = imread(path);
			imshow("Input Image", img);

			// ==================== STAGE 3: Filling the Graph =====================
			sprintf(path, "D:\\Res\\CNN 1024 Potentials\\t4-g%02d-%02d.dat", m, i);
			Mat pot = Serialize::from(path);
			for (int y = 0; y < pot.rows; y++) {
				float		* pPot = pot.ptr<float>(y);
				for (int x = 0; x < pot.cols; x++) {
					for (int s = 0; s < nStates; s++)
						pPot[x * nStates + s] = -logf(pPot[x * nStates + s]);
				} // x
			} // y

			DenseCRF2D crf(img.cols, img.rows, nStates);
			crf.setUnaryEnergy(reinterpret_cast<float *>(pot.data));
			crf.addPairwiseGaussian(3, 3, 3);
			crf.addPairwiseBilateral(60, 60, 20, 20, 20, img.data, 10);
			
			// ========================= STAGE 4: Decoding =========================
			short *map = new short[img.cols * img.rows];
			crf.map(10, map);
			vec_byte_t	optimalDecoding;
			for (int i = 0; i < img.cols * img.rows; i++)
				optimalDecoding.push_back( static_cast<byte>(map[i]));
			delete[] map;

			// ====================== Evaluation =======================
			Mat solution(img.size(), CV_8UC1, optimalDecoding.data());
			sprintf(path, "D:\\Res\\SC %d\\t4-g%02d-%02d-result.bmp", 1024, m, i);
			imwrite(path, solution);
			
			sprintf(path, "Z:\\Data\\_Kimiaki\\GroundTruthImages_BlackOrWhite\\t4-g%02d-%02d.png", m < 17 ? m : m - 1, i);
			Mat gt = imread(path, 0);
			imshow("Groundtruth", gt);
			gt /= 255;
			byte gt_state = (m > 17) ? m - 1 : m;
			gt *= gt_state;

			confMatGlobal.estimate(gt, solution);
			confMat.estimate(gt, solution);
			float accuracy = confMat.getAccuracy();
			confMat.reset();

			char str[255];
			sprintf(str, "Accuracy = %.2f%%", accuracy);
			printf("%s\n", str);

			// ====================== Visualization =======================
			Mat res = colorize<byte>(solution);
			imshow("Solution", res);
			sprintf(path, "D:\\Res\\SC %d\\t4-g%02d-%02d-solution-CNN_%d (%d,%02d).jpg", 1024, m, i, 1024, (int)accuracy, (int)((accuracy - (int)accuracy) * 100));
			imwrite(path, res);

			cvWaitKey(25);
		} // i
	} // m

	CMarker marker;
	Mat confusionMat = confMatGlobal.getConfusionMatrix();
	Mat confusionMatImg = marker.drawConfusionMatrix(confusionMat, MARK_BW | MARK_PERCLASS);
	imshow("Confusion Matrix", confusionMatImg);
	sprintf(path, "D:\\Res\\SC %d\\cMat.jpg", 1024);
	imwrite(path, confusionMatImg);

	cvWaitKey();
	return 0;

	/*
	CGraphExt graph(nStates);
	graph.build(pot.size());
	graph.setNodes(pot);
	vec_byte_t optimalDecoding = CDecode::decode(&graph);
	Mat solution(pot.size(), CV_8UC1, optimalDecoding.data());
	Mat resDGM = colorize<byte>(solution);
	imshow("Result DGB", resDGM);
	*/
}
