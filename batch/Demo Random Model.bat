if exist ..\Data\001_img.jpg if exist ..\Data\001_gt.bmp (
	"Demo Random Model.exe" 0 ..\Data\001_img.jpg ..\Data\001_gt.bmp .\output_RandomModel_Bayes.jpg
	"Demo Random Model.exe" 1 ..\Data\001_img.jpg ..\Data\001_gt.bmp .\output_RandomModel_GMM.jpg
	"Demo Random Model.exe" 2 ..\Data\001_img.jpg ..\Data\001_gt.bmp .\output_RandomModel_CvGMM.jpg
	"Demo Random Model.exe" 3 ..\Data\001_img.jpg ..\Data\001_gt.bmp .\output_RandomModel_KNN.jpg
	"Demo Random Model.exe" 4 ..\Data\001_img.jpg ..\Data\001_gt.bmp .\output_RandomModel_CvKNN.jpg	
	"Demo Random Model.exe" 5 ..\Data\001_img.jpg ..\Data\001_gt.bmp .\output_RandomModel_CvRF.jpg
	"Demo Random Model.exe" 6 ..\Data\001_img.jpg ..\Data\001_gt.bmp .\output_RandomModel_MsRF.jpg
	"Demo Random Model.exe" 7 ..\Data\001_img.jpg ..\Data\001_gt.bmp .\output_RandomModel_CvANN.jpg	
	"Demo Random Model.exe" 8 ..\Data\001_img.jpg ..\Data\001_gt.bmp .\output_RandomModel_CvSVM.jpg
)
