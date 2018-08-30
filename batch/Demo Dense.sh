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

	"./Demo Dense" ../data/001_fv.jpg ../data/001_gt.bmp ../data/002_fv.jpg ../data/002_gt.bmp ../data/002_img.jpg ./output_denseCRF.jpg
fi
