if exist ..\Data\001_img.jpg if exist ..\Data\001_gt.bmp if exist ..\Data\002_img.jpg if exist ..\Data\002_gt.bmp (
	if not exist ..\Data\001_fv.jpg (
		"Demo Feature Extraction.exe" ..\Data\001_img.jpg ..\Data\001_fv.jpg
	)
	if not exist ..\Data\002_fv.jpg (
		"Demo Feature Extraction.exe" ..\Data\002_img.jpg ..\Data\002_fv.jpg
	)
	"Demo Train.exe" 0 0 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Bayes_NoEdges.jpg
	"Demo Train.exe" 1 0 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_GMM_NoEdges.jpg
	"Demo Train.exe" 2 0 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_CvGMM_NoEdges.jpg
	"Demo Train.exe" 3 0 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_KNN_NoEdges.jpg
	"Demo Train.exe" 4 0 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_CvRF_NoEdges.jpg
	"Demo Train.exe" 5 0 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_MsRF_NoEdges.jpg
                                                                                                                         
	"Demo Train.exe" 0 1 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Bayes_Potts.jpg
	"Demo Train.exe" 1 1 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_GMM_Potts.jpg
	"Demo Train.exe" 2 1 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_CvGMM_Potts.jpg
	"Demo Train.exe" 3 0 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_KNN_Potts.jpg
	"Demo Train.exe" 4 1 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_CvRF_Potts.jpg
	"Demo Train.exe" 5 1 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_MsRF_Potts.jpg
                                                                                                                         
	"Demo Train.exe" 0 2 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Bayes_PottsCS.jpg
	"Demo Train.exe" 1 2 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_GMM_PottsCS.jpg
	"Demo Train.exe" 2 2 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_CvGMM_PottsCS.jpg
	"Demo Train.exe" 3 0 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_KNN_PottsCS.jpg
	"Demo Train.exe" 4 2 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_CvRF_PottsCS.jpg
	"Demo Train.exe" 5 2 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_MsRF_PottsCS.jpg
                                                                                                                         
	"Demo Train.exe" 0 3 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Bayes_Prior.jpg
	"Demo Train.exe" 1 3 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_GMM_Prior.jpg
	"Demo Train.exe" 2 3 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_CvGMM_Prior.jpg
	"Demo Train.exe" 3 0 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_KNN_Prior.jpg
	"Demo Train.exe" 4 3 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_CvRF_Prior.jpg
	"Demo Train.exe" 5 3 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_MsRF_Prior.jpg
                                                                                                                         
	"Demo Train.exe" 0 4 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_Bayes_Concat.jpg
	"Demo Train.exe" 1 4 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_GMM_Concat.jpg
	"Demo Train.exe" 2 4 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_CvGMM_Concat.jpg
	"Demo Train.exe" 3 0 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_KNN_Concat.jpg
	"Demo Train.exe" 4 4 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_CvRF_Concat.jpg
	"Demo Train.exe" 5 4 ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg .\output_MsRF_Concat.jpg
)
