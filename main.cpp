#include <cv.h>
#include <cvaux.h>
#include <highgui.h>
#include <iostream>

#include "cvTracker.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
	Mat img1;
	CvTracker tracker;
	char img_file[20] = "../seqG/0001.png";

	namedWindow("Win", 1);

	tracker.priorImage(imread(img_file, CV_LOAD_IMAGE_COLOR));
	tracker.addROI(Rect(135, 100, 50, 50));

	for(int i = 0; i < 1000; i++) {
		sprintf(img_file, "../seqG/%04d.png", i);

		img1 = imread(img_file, CV_LOAD_IMAGE_COLOR);
		if(!img1.empty()) {

			img1 = tracker.updateTracks(img1);

			imshow("Win", img1);
			waitKey(30);
		}
	}

	return 0;
}

