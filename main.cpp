#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "cvTracker.h"

using namespace std;
using namespace cv;

int main() {

	//CvTracker* tracker = new CvTracker();
	//tracker->addObject(...);
	Mat img;
	img = imread("ex.jpg", CV_LOAD_IMAGE_UNCHANGED);

	return 0;
}
