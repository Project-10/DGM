1f exist ..\Data\001_img.jpg if exist ..\Data\001_gt.bmp if exist ..\Data\002_img.jpg if exist ..\Data\002_gt.bmp (
	if not exist ..\Data\001_fv.jpg (
		"Demo Feature Extraction.exe" ..\Data\001_img.jpg ..\Data\001_fv.jpg
	)
	if not exist ..\Data\002_fv.jpg (
		"Demo Feature Extraction.exe" ..\Data\002_img.jpg ..\Data\002_fv.jpg
	)
	"Demo Train.exe" 0 0 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Train_Bayes_NoEdges.jpg
	"Demo Train.exe" 1 0 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Train_GMM_NoEdges.jpg
	"Demo Train.exe" 2 0 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Train_CvGMM_NoEdges.jpg
	"Demo Train.exe" 3 0 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Train_KNN_NoEdges.jpg
	"Demo Train.exe" 4 0 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Train_CvKNN_NoEdges.jpg
	"Demo Train.exe" 5 0 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Train_CvRF_NoEdges.jpg
	"Demo Train.exe" 6 0 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Train_MsRF_NoEdges.jpg
	"Demo Train.exe" 7 0 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Train_CvANN_NoEdges.jpg	
	"Demo Train.exe" 8 0 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Train_CvSVM_NoEdges.jpg		
                                                                                                                         
	"Demo Train.exe" 0 1 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Train_Bayes_Potts.jpg
	"Demo Train.exe" 1 1 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Train_GMM_Potts.jpg
	"Demo Train.exe" 2 1 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Train_CvGMM_Potts.jpg
	"Demo Train.exe" 3 1 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Train_KNN_Potts.jpg
	"Demo Train.exe" 4 1 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Train_CvKNN_Potts.jpg
	"Demo Train.exe" 5 1 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Train_CvRF_Potts.jpg
	"Demo Train.exe" 6 1 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Train_MsRF_Potts.jpg
	"Demo Train.exe" 7 1 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Train_CvANN_Potts.jpg	
	"Demo Train.exe" 8 1 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Train_CvSVM_Potts.jpg	
	
	"Demo Train.exe" 0 2 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Train_Bayes_PottsCS.jpg
	"Demo Train.exe" 1 2 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Train_GMM_PottsCS.jpg
	"Demo Train.exe" 2 2 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Train_CvGMM_PottsCS.jpg
	"Demo Train.exe" 3 2 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Train_KNN_PottsCS.jpg
	"Demo Train.exe" 4 2 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Train_CvKNN_PottsCS.jpg
	"Demo Train.exe" 5 2 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Train_CvRF_PottsCS.jpg
	"Demo Train.exe" 6 2 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Train_MsRF_PottsCS.jpg
	"Demo Train.exe" 7 2 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Train_CvANN_PottsCS.jpg	
	"Demo Train.exe" 8 2 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Train_CvSVM_PottsCS.jpg	
	
	"Demo Train.exe" 0 3 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Train_Bayes_Prior.jpg
	"Demo Train.exe" 1 3 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Train_GMM_Prior.jpg
	"Demo Train.exe" 2 3 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Train_CvGMM_Prior.jpg
	"Demo Train.exe" 3 3 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Train_KNN_Prior.jpg
	"Demo Train.exe" 4 3 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Train_CvKNN_Prior.jpg
	"Demo Train.exe" 5 3 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Train_CvRF_Prior.jpg
	"Demo Train.exe" 6 3 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Train_MsRF_Prior.jpg
 	"Demo Train.exe" 7 3 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Train_CvANN_Prior.jpg	
	"Demo Train.exe" 8 3 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Train_CvSVM_Prior.jpg	                                                                                                                        
																														 
	"Demo Train.exe" 0 4 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Train_Bayes_Concat.jpg
	"Demo Train.exe" 1 4 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Train_GMM_Concat.jpg
	"Demo Train.exe" 2 4 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Train_CvGMM_Concat.jpg
	"Demo Train.exe" 3 4 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Train_KNN_Concat.jpg
	"Demo Train.exe" 4 4 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Train_CvKNN_Concat.jpg
	"Demo Train.exe" 5 4 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Train_CvRF_Concat.jpg
	"Demo Train.exe" 6 4 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Train_MsRF_Concat.jpg
	"Demo Train.exe" 7 4 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Train_CvANN_Concat.jpg	
	"Demo Train.exe" 8 4 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Train_CvSVM_Concat.jpg		
)
