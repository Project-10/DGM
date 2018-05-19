if [ -e ../data/001_img.jpg ] && [ -e ../data/001_gt.bmp ]
then
	"./Demo Random Model" 0 ../data/001_img.jpg ../data/001_gt.bmp ./output_RandomModel_Bayes.jpg
	"./Demo Random Model" 1 ../data/001_img.jpg ../data/001_gt.bmp ./output_RandomModel_GMM.jpg
	"./Demo Random Model" 2 ../data/001_img.jpg ../data/001_gt.bmp ./output_RandomModel_CvGMM.jpg
	"./Demo Random Model" 3 ../data/001_img.jpg ../data/001_gt.bmp ./output_RandomModel_KNN.jpg
	"./Demo Random Model" 4 ../data/001_img.jpg ../data/001_gt.bmp ./output_RandomModel_CvKNN.jpg	
	"./Demo Random Model" 5 ../data/001_img.jpg ../data/001_gt.bmp ./output_RandomModel_CvRF.jpg
	"./Demo Random Model" 6 ../data/001_img.jpg ../data/001_gt.bmp ./output_RandomModel_MsRF.jpg
	"./Demo Random Model" 7 ../data/001_img.jpg ../data/001_gt.bmp ./output_RandomModel_CvANN.jpg	
	"./Demo Random Model" 8 ../data/001_img.jpg ../data/001_gt.bmp ./output_RandomModel_CvSVM.jpg
fi
