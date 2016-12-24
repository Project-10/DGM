@echo off
SET version=320
SET OpenCV_src=%OPENCVDIR%\build\x64\vc14\bin

@echo on
copy %OpenCV_src%\opencv_core%version%.dll .\
copy %OpenCV_src%\opencv_features2d%version%.dll .\
copy %OpenCV_src%\opencv_flann%version%.dll .\
copy %OpenCV_src%\opencv_highgui%version%.dll .\
copy %OpenCV_src%\opencv_imgproc%version%.dll .\
copy %OpenCV_src%\opencv_imgcodecs%version%.dll .\
copy %OpenCV_src%\opencv_videoio%version%.dll .\
copy %OpenCV_src%\opencv_ml%version%.dll .\
                
copy %OpenCV_src%\opencv_core%version%d.dll .\
copy %OpenCV_src%\opencv_features2d%version%d.dll .\
copy %OpenCV_src%\opencv_flann%version%d.dll .\
copy %OpenCV_src%\opencv_highgui%version%d.dll .\
copy %OpenCV_src%\opencv_imgproc%version%d.dll .\
copy %OpenCV_src%\opencv_imgcodecs%version%d.dll .\
copy %OpenCV_src%\opencv_videoio%version%d.dll .\
copy %OpenCV_src%\opencv_ml%version%d.dll .\
