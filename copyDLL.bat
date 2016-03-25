@echo off
SET version=310

@echo on
copy %OPENCVDIR%\build\x86\vc14\bin\opencv_core%version%.dll .\bin\Release
copy %OPENCVDIR%\build\x86\vc14\bin\opencv_highgui%version%.dll .\bin\Release
copy %OPENCVDIR%\build\x86\vc14\bin\opencv_imgproc%version%.dll .\bin\Release
copy %OPENCVDIR%\build\x86\vc14\bin\opencv_imgcodecs%version%.dll .\bin\Release
copy %OPENCVDIR%\build\x86\vc14\bin\opencv_videoio%version%.dll .\bin\Release
copy %OPENCVDIR%\build\x86\vc14\bin\opencv_ml%version%.dll .\bin\Release

copy %OPENCVDIR%\build\x86\vc14\bin\opencv_core%version%d.dll .\bin\Debug
copy %OPENCVDIR%\build\x86\vc14\bin\opencv_highgui%version%d.dll .\bin\Debug
copy %OPENCVDIR%\build\x86\vc14\bin\opencv_imgproc%version%d.dll .\bin\Debug
copy %OPENCVDIR%\build\x86\vc14\bin\opencv_imgcodecs%version%d.dll .\bin\Debug
copy %OPENCVDIR%\build\x86\vc14\bin\opencv_videoio%version%d.dll .\bin\Debug
copy %OPENCVDIR%\build\x86\vc14\bin\opencv_ml%version%d.dll .\bin\Debug

copy %OPENCVDIR%\build\x64\vc14\bin\opencv_core%version%.dll .\bin\x64\Release
copy %OPENCVDIR%\build\x64\vc14\bin\opencv_highgui%version%.dll .\bin\x64\Release
copy %OPENCVDIR%\build\x64\vc14\bin\opencv_imgproc%version%.dll .\bin\x64\Release
copy %OPENCVDIR%\build\x64\vc14\bin\opencv_imgcodecs%version%.dll .\bin\x64\Release
copy %OPENCVDIR%\build\x64\vc14\bin\opencv_videoio%version%.dll .\bin\x64\Release
copy %OPENCVDIR%\build\x64\vc14\bin\opencv_ml%version%.dll .\bin\x64\Release

copy %OPENCVDIR%\build\x64\vc14\bin\opencv_core%version%d.dll .\bin\x64\Debug
copy %OPENCVDIR%\build\x64\vc14\bin\opencv_highgui%version%d.dll .\bin\x64\Debug
copy %OPENCVDIR%\build\x64\vc14\bin\opencv_imgproc%version%d.dll .\bin\x64\Debug
copy %OPENCVDIR%\build\x64\vc14\bin\opencv_imgcodecs%version%d.dll .\bin\x64\Debug
copy %OPENCVDIR%\build\x64\vc14\bin\opencv_videoio%version%d.dll .\bin\x64\Debug
copy %OPENCVDIR%\build\x64\vc14\bin\opencv_ml%version%d.dll .\bin\x64\Debug
