#!/bin/bash
if [ -e ../data/001_img.jpg ] && [ -e ../data/001_gt.bmp ] && [ -e ../data/002_img.jpg ] && [ -e ../data/002_gt.bmp ]
then
	if [ ! -e ../data/001_fv.jpg ]
	then
		"./Demo Feature Extraction" ../data/001_img.jpg ../data/001_fv.jpg
	fi
	if [ ! -e ../data/002_fv.jpg ]
	then
		"./Demo Feature Extraction" ../data/002_img.jpg ../data/002_fv.jpg
	fi

	"./Demo Train" 0 0 ../data/001_fv.jpg ../data/001_gt.bmp ../data/002_fv.jpg ../data/002_gt.bmp ../data/002_img.jpg ./output_Train_Bayes_NoEdges.jpg
	"./Demo Train" 1 0 ../data/001_fv.jpg ../data/001_gt.bmp ../data/002_fv.jpg ../data/002_gt.bmp ../data/002_img.jpg ./output_Train_GMM_NoEdges.jpg
	"./Demo Train" 2 0 ../data/001_fv.jpg ../data/001_gt.bmp ../data/002_fv.jpg ../data/002_gt.bmp ../data/002_img.jpg ./output_Train_CvGMM_NoEdges.jpg
	"./Demo Train" 3 0 ../data/001_fv.jpg ../data/001_gt.bmp ../data/002_fv.jpg ../data/002_gt.bmp ../data/002_img.jpg ./output_Train_KNN_NoEdges.jpg
	"./Demo Train" 4 0 ../data/001_fv.jpg ../data/001_gt.bmp ../data/002_fv.jpg ../data/002_gt.bmp ../data/002_img.jpg ./output_Train_CvKNN_NoEdges.jpg
	"./Demo Train" 5 0 ../data/001_fv.jpg ../data/001_gt.bmp ../data/002_fv.jpg ../data/002_gt.bmp ../data/002_img.jpg ./output_Train_CvRF_NoEdges.jpg
	"./Demo Train" 6 0 ../data/001_fv.jpg ../data/001_gt.bmp ../data/002_fv.jpg ../data/002_gt.bmp ../data/002_img.jpg ./output_Train_MsRF_NoEdges.jpg
	"./Demo Train" 7 0 ../data/001_fv.jpg ../data/001_gt.bmp ../data/002_fv.jpg ../data/002_gt.bmp ../data/002_img.jpg ./output_Train_CvANN_NoEdges.jpg	
	"./Demo Train" 8 0 ../data/001_fv.jpg ../data/001_gt.bmp ../data/002_fv.jpg ../data/002_gt.bmp ../data/002_img.jpg ./output_Train_CvSVM_NoEdges.jpg		
     
	"./Demo Train" 0 1 ../data/001_fv.jpg ../data/001_gt.bmp ../data/002_fv.jpg ../data/002_gt.bmp ../data/002_img.jpg ./output_Train_Bayes_Potts.jpg
	"./Demo Train" 1 1 ../data/001_fv.jpg ../data/001_gt.bmp ../data/002_fv.jpg ../data/002_gt.bmp ../data/002_img.jpg ./output_Train_GMM_Potts.jpg
	"./Demo Train" 2 1 ../data/001_fv.jpg ../data/001_gt.bmp ../data/002_fv.jpg ../data/002_gt.bmp ../data/002_img.jpg ./output_Train_CvGMM_Potts.jpg
	"./Demo Train" 3 1 ../data/001_fv.jpg ../data/001_gt.bmp ../data/002_fv.jpg ../data/002_gt.bmp ../data/002_img.jpg ./output_Train_KNN_Potts.jpg
	"./Demo Train" 4 1 ../data/001_fv.jpg ../data/001_gt.bmp ../data/002_fv.jpg ../data/002_gt.bmp ../data/002_img.jpg ./output_Train_CvKNN_Potts.jpg
	"./Demo Train" 5 1 ../data/001_fv.jpg ../data/001_gt.bmp ../data/002_fv.jpg ../data/002_gt.bmp ../data/002_img.jpg ./output_Train_CvRF_Potts.jpg
	"./Demo Train" 6 1 ../data/001_fv.jpg ../data/001_gt.bmp ../data/002_fv.jpg ../data/002_gt.bmp ../data/002_img.jpg ./output_Train_MsRF_Potts.jpg
	"./Demo Train" 7 1 ../data/001_fv.jpg ../data/001_gt.bmp ../data/002_fv.jpg ../data/002_gt.bmp ../data/002_img.jpg ./output_Train_CvANN_Potts.jpg	
	"./Demo Train" 8 1 ../data/001_fv.jpg ../data/001_gt.bmp ../data/002_fv.jpg ../data/002_gt.bmp ../data/002_img.jpg ./output_Train_CvSVM_Potts.jpg	

	"./Demo Train" 0 2 ../data/001_fv.jpg ../data/001_gt.bmp ../data/002_fv.jpg ../data/002_gt.bmp ../data/002_img.jpg ./output_Train_Bayes_PottsCS.jpg
	"./Demo Train" 1 2 ../data/001_fv.jpg ../data/001_gt.bmp ../data/002_fv.jpg ../data/002_gt.bmp ../data/002_img.jpg ./output_Train_GMM_PottsCS.jpg
	"./Demo Train" 2 2 ../data/001_fv.jpg ../data/001_gt.bmp ../data/002_fv.jpg ../data/002_gt.bmp ../data/002_img.jpg ./output_Train_CvGMM_PottsCS.jpg
	"./Demo Train" 3 2 ../data/001_fv.jpg ../data/001_gt.bmp ../data/002_fv.jpg ../data/002_gt.bmp ../data/002_img.jpg ./output_Train_KNN_PottsCS.jpg
	"./Demo Train" 4 2 ../data/001_fv.jpg ../data/001_gt.bmp ../data/002_fv.jpg ../data/002_gt.bmp ../data/002_img.jpg ./output_Train_CvKNN_PottsCS.jpg
	"./Demo Train" 5 2 ../data/001_fv.jpg ../data/001_gt.bmp ../data/002_fv.jpg ../data/002_gt.bmp ../data/002_img.jpg ./output_Train_CvRF_PottsCS.jpg
	"./Demo Train" 6 2 ../data/001_fv.jpg ../data/001_gt.bmp ../data/002_fv.jpg ../data/002_gt.bmp ../data/002_img.jpg ./output_Train_MsRF_PottsCS.jpg
	"./Demo Train" 7 2 ../data/001_fv.jpg ../data/001_gt.bmp ../data/002_fv.jpg ../data/002_gt.bmp ../data/002_img.jpg ./output_Train_CvANN_PottsCS.jpg	
	"./Demo Train" 8 2 ../data/001_fv.jpg ../data/001_gt.bmp ../data/002_fv.jpg ../data/002_gt.bmp ../data/002_img.jpg ./output_Train_CvSVM_PottsCS.jpg	
	
	"./Demo Train" 0 3 ../data/001_fv.jpg ../data/001_gt.bmp ../data/002_fv.jpg ../data/002_gt.bmp ../data/002_img.jpg ./output_Train_Bayes_Prior.jpg
	"./Demo Train" 1 3 ../data/001_fv.jpg ../data/001_gt.bmp ../data/002_fv.jpg ../data/002_gt.bmp ../data/002_img.jpg ./output_Train_GMM_Prior.jpg
	"./Demo Train" 2 3 ../data/001_fv.jpg ../data/001_gt.bmp ../data/002_fv.jpg ../data/002_gt.bmp ../data/002_img.jpg ./output_Train_CvGMM_Prior.jpg
	"./Demo Train" 3 3 ../data/001_fv.jpg ../data/001_gt.bmp ../data/002_fv.jpg ../data/002_gt.bmp ../data/002_img.jpg ./output_Train_KNN_Prior.jpg
	"./Demo Train" 4 3 ../data/001_fv.jpg ../data/001_gt.bmp ../data/002_fv.jpg ../data/002_gt.bmp ../data/002_img.jpg ./output_Train_CvKNN_Prior.jpg
	"./Demo Train" 5 3 ../data/001_fv.jpg ../data/001_gt.bmp ../data/002_fv.jpg ../data/002_gt.bmp ../data/002_img.jpg ./output_Train_CvRF_Prior.jpg
	"./Demo Train" 6 3 ../data/001_fv.jpg ../data/001_gt.bmp ../data/002_fv.jpg ../data/002_gt.bmp ../data/002_img.jpg ./output_Train_MsRF_Prior.jpg
 	"./Demo Train" 7 3 ../data/001_fv.jpg ../data/001_gt.bmp ../data/002_fv.jpg ../data/002_gt.bmp ../data/002_img.jpg ./output_Train_CvANN_Prior.jpg	
	"./Demo Train" 8 3 ../data/001_fv.jpg ../data/001_gt.bmp ../data/002_fv.jpg ../data/002_gt.bmp ../data/002_img.jpg ./output_Train_CvSVM_Prior.jpg	                                                                                                                        

	"./Demo T/din"/0 4 ../data/0/d_fv/jpg ../data/0/d_gt/bmp ../data/0/d_fv/jpg ../data/0/2_gt.bmp ../data/002_img.jpg ./output_Train_Bayes_Concat.jpg
	"./Demo Train" 1 4 ../data/001_fv.jpg ../data/001_gt.bmp ../data/002_fv.jpg ../data/002_gt.bmp ../data/002_img.jpg ./output_Train_GMM_Concat.jpg
	"./Demo Train" 2 4 ../data/001_fv.jpg ../data/001_gt.bmp ../data/002_fv.jpg ../data/002_gt.bmp ../data/002_img.jpg ./output_Train_CvGMM_Concat.jpg
	"./Demo Train" 3 4 ../data/001_fv.jpg ../data/001_gt.bmp ../data/002_fv.jpg ../data/002_gt.bmp ../data/002_img.jpg ./output_Train_KNN_Concat.jpg
	"./Demo Train" 4 4 ../data/001_fv.jpg ../data/001_gt.bmp ../data/002_fv.jpg ../data/002_gt.bmp ../data/002_img.jpg ./output_Train_CvKNN_Concat.jpg
	"./Demo Train" 5 4 ../data/001_fv.jpg ../data/001_gt.bmp ../data/002_fv.jpg ../data/002_gt.bmp ../data/002_img.jpg ./output_Train_CvRF_Concat.jpg
	"./Demo Train" 6 4 ../data/001_fv.jpg ../data/001_gt.bmp ../data/002_fv.jpg ../data/002_gt.bmp ../data/002_img.jpg ./output_Train_MsRF_Concat.jpg
	"./Demo Train" 7 4 ../data/001_fv.jpg ../data/001_gt.bmp ../data/002_fv.jpg ../data/002_gt.bmp ../data/002_img.jpg ./output_Train_CvANN_Concat.jpg	
	"./Demo Train" 8 4 ../data/001_fv.jpg ../data/001_gt.bmp ../data/002_fv.jpg ../data/002_gt.bmp ../data/002_img.jpg ./output_Train_CvSVM_Concat.jpg		
fi
