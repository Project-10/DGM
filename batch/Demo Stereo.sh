if [ -e ../data/001_img.jpg ] && [ -e ../data/001_fv.jpg ] && [ -e ../data/001_gt.bmp ]
then 
	"./Demo Stereo" ../data/tsukuba_left.jpg ../data/tsukuba_right.jpg 5 16
fi
	