if exist ..\Data\001_img.jpg if exist ..\Data\001_gt.bmp (
	"Demo Random Model.exe" 0 ..\Data\001_img.jpg ..\Data\001_gt.bmp .\output_Bayes.jpg
	"Demo Random Model.exe" 1 ..\Data\001_img.jpg ..\Data\001_gt.bmp .\output_GMM.jpg
	"Demo Random Model.exe" 2 ..\Data\001_img.jpg ..\Data\001_gt.bmp .\output_CvGMM.jpg
	"Demo Random Model.exe" 3 ..\Data\001_img.jpg ..\Data\001_gt.bmp .\output_KNN.jpg
	"Demo Random Model.exe" 4 ..\Data\001_img.jpg ..\Data\001_gt.bmp .\output_CvRF.jpg
	"Demo Random Model.exe" 5 ..\Data\001_img.jpg ..\Data\001_gt.bmp .\output_MsRF.jpg                                                                                                                         
)
