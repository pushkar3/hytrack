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
        ftracker.init(image, selection);
        trackObject = 1;
        break;
    }
}

float w1 = 0.5;
float w2 = 0.5;
// TODO: EM!

int main(int argc, char** argv) {
	char img_file[20] = "seqG/0001.png";

	namedWindow("Win", 1);
	setMouseCallback("Win", onMouse, 0);

	for(int i = 0; i < 1000; i++) {
		sprintf(img_file, "seqG/%04d.png", i);
		image = imread(img_file, CV_LOAD_IMAGE_COLOR);

		if (!image.empty()) {

			if(trackObject) {
				Rect win1 = ctracker.track(image);
				rectangle(image, Point(win1.x, win1.y), Point(win1.x + win1.width, win1.y + win1.height), Scalar(0, 0, 255), 2, CV_AA);
				Rect win2 = ftracker.track(image);
				rectangle(image, Point(win2.x, win2.y), Point(win2.x + win2.width, win2.y + win2.height), Scalar(0, 255, 0), 2, CV_AA);
				Rect win3 = win1;
				win3.x = w1*win1.x + w2*win2.x;
				win3.y = w1*win1.y + w2*win2.y;
				rectangle(image, Point(win3.x, win3.y), Point(win3.x + win3.width, win3.y + win3.height), Scalar(255, 0, 0), 2, CV_AA);
			}

			if (selectObject && selection.width > 0 && selection.height > 0) {
				Mat roi(image, selection);
				bitwise_not(roi, roi);
			}


			imshow("Win", image);
			waitKey(30);
		}
		else
			i =0;
	}

	return 0;
}

