// Example Feature Extraction
#include "FEX.h"

using namespace DirectGraphicalModels;

void print_help(void)
{
	printf("Usage: \"Demo Feature Extraction.exe\" input_image output_image\n");
}

int main(int argc, char *argv[])
{
	if (argc != 3) {
		print_help();
		return 0;
	}

	Mat img = imread(argv[1], 1);
	fex::CCommonFeatureExtractor fExtractor(img);

	Mat coord = fex::CCoordinate::get(img);

	// Extracting 3 features
	Mat ndvi		= fExtractor.getNDVI(10).get();											// NDVI feature
	Mat variance	= fExtractor.getIntensity(CV_RGB(0.0, 0.5, 0.5)).getVariance().get();	// Variance of intensity feature
	Mat saturation	= fExtractor.getSaturation().invert().get();							// Inverted saturation feature

	// Storing 3 features in a 3 channel RGB image
	Mat			featureImg;
	vec_mat_t	channels;
	channels.push_back(ndvi);			// blue channel
	channels.push_back(variance);		// green channel
	channels.push_back(saturation);		// red channel
	merge(channels, featureImg);

	imwrite(argv[2], featureImg);
	return 0;
}
