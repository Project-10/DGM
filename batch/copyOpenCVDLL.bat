@echo off
SET version=2413
SET OpenCV_src=%OPENCVDIR%\build\x64\vc14\bin

@echo on
copy %OpenCV_src%\opencv_core%version%.dll .\
copy %OpenCV_src%\opencv_highgui%version%.dll .\
copy %OpenCV_src%\opencv_imgproc%version%.dll .\
copy %OpenCV_src%\opencv_ml%version%.dll .\
                
copy %OpenCV_src%\opencv_core%version%d.dll .\
copy %OpenCV_src%\opencv_highgui%version%d.dll .\
copy %OpenCV_src%\opencv_imgproc%version%d.dll .\
copy %OpenCV_src%\opencv_ml%version%d.dll .\
