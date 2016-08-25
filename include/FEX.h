#pragma once

#include "FEX/CommonFeatureExtractor.h"
#include "FEX/SparseDictionary.h"

/**
@defgroup moduleFEX FEX Module
@section sec_fex_title Feature Extraction Module

allows for extracting various descriptors from images, which are useful for classification. 

For the practical application, the original input data is preprocessed to transform it into some new space of descriptors (features) where, it is hoped, the classification problem will be easier to solve.
The main idea of this preprocessing is to reduce the variability of input data for each class, and thus to make it much easier for a subsequent classification algorithm to distinguish between the
different classes. This preprocessing stage is also called feature extraction. Note that new test data must be preprocessed using the same steps as the training data.


 Method               | Class                                     | Input              | Output
 -------------------- | ----------------------------------------- | :----------------: | :-----:
 Coordinate           | DirectGraphicalModels::fex::CCoordinate   | any                | CV_8UC1
 Distance             | DirectGraphicalModels::fex::CDistance	  | CV_8UC1 or CV_8UC3 | CV_8UC1
 Gradient             | DirectGraphicalModels::fex::CGradient	  | CV_8UC1 or CV_8UC3 | CV_8UC1
 HOG                  | DirectGraphicalModels::fex::CHOG		  | CV_8UC1 or CV_8UC3 | CV_8UC{nBins}
 Intensity            | DirectGraphicalModels::fex::CIntensity    | CV_8UC3            | CV_8UC1
 NDVI                 | DirectGraphicalModels::fex::CNDVI		  | CV_8UC3            | CV_8UC1
 Hue-Saturation-Value | DirectGraphicalModels::fex::CHSV          | CV_8UC3            | CV_8UC3
 Scale                | DirectGraphicalModels::fex::CScale	      | any                | CV_8UC1
 Variance             | DirectGraphicalModels::fex::CVariance	  | CV_8UC1 or CV_8UC3 | CV_8UC1
 Sparse Coding        | DirectGraphicalModels::fex::CSparseCoding | CV_8UC1 or CV_8UC3 | CV_8UC{nWords}

There are 3 ways of using the feature extraction module in your code. Let us cosider the extraction of \a coordinate feature.

The first way is to declare the correspondig class and call its method DirectGraphicalModels::fex::IFeatureExtractor::get() :
@code
using namespace DirectGraphicalModels::fex;

CCoordinate cfExtractor(img)
Mat coordinatate = cfExtractor.get();
@endcode

Alternatively one may call the corresponding static function, without declring the class instance:
@code
using namespace DirectGraphicalModels::fex;

Mat coordinatate = CCoordinate::get(img);
@endcode

The third way is to use the common feature extracton interface DirectGraphicalModels::fex::CCommonFeatureExtractor, which supports <a href="https://en.wikipedia.org/wiki/Fluent_interface">fluent interface</a>. 
Please see also the class documentation for more details:
@code
using namespace DirectGraphicalModels::fex;

CCommonFeatureExtractor fExtractor(img);
Mat coordinatate = fExtractor.getCoordinate().get();	
@endcode

Please see also our tutorial: @ref demofex.

@author Sergey G. Kosov, sergey.kosov@project-10.de
*/

/**
@page demofex Demo Feature Extraction
In this example we extract 3 features from the input image <b>Original Image.jpg</b>. These features are <i>NDVI, Variance of intensity</i> and <i>saturation</i>. 
The features are calculated for every pixel of the input image and thus are represented as images of the same resolution as the input one. It is often convenient to have a set of 
features in form of one multi-channel image. For storing such a multi-channel image is usually split into a number of 3-channels RGB images. Hence, for sake of simplicity,
we extract only 3 features in this example.

<table align="center">
<tr>
	<td><center><b>Input</b></center></td>
	<td></td>
	<td><center><b>Output</b></center></td>
</tr>
<tr>
  <td><img src="001_img_small.jpg"></td>
  <td><img src="arrow.png"></td>
  <td><img src="001_fv_small.jpg"></td>
</tr>
<tr>
  <td><center><b>Original Image.jpg</b></center></td>
  <td></td>
  <td><center><b>Resulting Feature Vector</b></center></td>
</tr>
</table>

@code
#include "FEX.h"
using namespace DirectGraphicalModels;

int main()
{
	Mat img = imread("Original Image.jpg", 1);
	fex::CCommonFeatureExtractor fExtractor(img);

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

	imshow("Feature Vector", featureImg);
	cvWaitKey();
	return 0;
}
@endcode

*/

