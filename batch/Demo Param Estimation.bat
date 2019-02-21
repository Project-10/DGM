if exist ..\Data\001_img.jpg if exist ..\Data\001_gt.bmp if exist ..\Data\002_img.jpg if exist ..\Data\002_gt.bmp (
	if not exist ..\Data\001_fv.jpg (
		"Demo Feature Extraction.exe" ..\Data\001_img.jpg ..\Data\001_fv.jpg
	)
	if not exist ..\Data\002_fv.jpg (
		"Demo Feature Extraction.exe" ..\Data\002_img.jpg ..\Data\002_fv.jpg
	)
	"Demo Param Estimation.exe" ..\Data\001_fv.jpg ..\Data\001_gt.bmp ..\Data\002_fv.jpg ..\Data\002_gt.bmp ..\Data\002_img.jpg
)
