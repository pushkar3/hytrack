#include <cv.h>
#include <cvaux.h>
#include <highgui.h>
#include <iostream>

#include "colortracker.h"
#include "featuretracker.h"

using namespace cv;
using namespace std;

Mat image;
Rect selection;
Point origin;
bool selectObject = false;
int trackObject = 0;

ColorTracker ctracker;
FeatureTracker ftracker;

void onMouse(int event, int x, int y, int, void*) {
    if(selectObject) {
        selection.x = MIN(x, origin.x);
        selection.y = MIN(y, origin.y);
        selection.width = std::abs(x - origin.x);
        selection.height = std::abs(y - origin.y);
        selection &= Rect(0, 0, image.cols, image.rows);
    }

    switch(event) {
    case CV_EVENT_LBUTTONDOWN:
        origin = Point(x,y);
        selection = Rect(x,y,0,0);
        selectObject = true;
        break;
    case CV_EVENT_LBUTTONUP:
        selectObject = false;
        ctracker.init(image, selection);
        trackObject = 1;
        break;
    }
}


int main(int argc, char** argv) {
	char img_file[20] = "seqG/0001.png";

	namedWindow("Win", 1);
	setMouseCallback("Win", onMouse, 0);

	for(int i = 0; i < 1000; i++) {
		sprintf(img_file, "../seqG/%04d.png", i);
		image = imread(img_file, CV_LOAD_IMAGE_COLOR);

		if (!image.empty()) {

			if(trackObject) {
				Rect win = ctracker.track(image);
				rectangle(image, Point(win.x, win.y), Point(win.x + win.width, win.y + win.height), Scalar(0, 0, 255), 3, CV_AA);
			}

			if (selectObject && selection.width > 0 && selection.height > 0) {
				Mat roi(image, selection);
				bitwise_not(roi, roi);
			}

			imshow("Win", image);
			waitKey(30);
		}
	}

	return 0;
}

