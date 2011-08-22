#include <cv.h>
#include <ml.h>
#include <cvaux.h>
#include <highgui.h>
#include <stdio.h>
#include <time.h>
#include <iostream>

#include "opencv2/tracker/hybridtracker.hpp"

using namespace cv;
using namespace std;

Mat image;
Rect selection;
Point origin;
bool selectObject = false;
int trackObject = 0;

void drawRectangle(Mat* image, Rect win) {
	rectangle(*image, Point(win.x, win.y), Point(win.x + win.width, win.y
			+ win.height), Scalar(0, 255, 0), 2, CV_AA);
}

void onMouse(int event, int x, int y, int, void*) {
	if (selectObject) {
		selection.x = MIN(x, origin.x);
		selection.y = MIN(y, origin.y);
		selection.width = std::abs(x - origin.x);
		selection.height = std::abs(y - origin.y);
		selection &= Rect(0, 0, image.cols, image.rows);
	}

	switch (event) {
	case CV_EVENT_LBUTTONDOWN:
		origin = Point(x, y);
		selection = Rect(x, y, 0, 0);
		selectObject = true;
		break;
	case CV_EVENT_LBUTTONUP:
		selectObject = false;
		trackObject = -1;
		cout << "Init done" << endl;
		break;
	}
}



int main(int argc, char** argv) {

	HybridTrackerParams params;
	HybridTracker tracker(params);

	char img_file[20] = "seqG/0001.png";
	namedWindow("Win", 1);
	setMouseCallback("Win", onMouse, 0);


	for (int i = 0; i < 1000; i++) {

		sprintf(img_file, "seqG/%04d.png", i);
		image = imread(img_file, CV_LOAD_IMAGE_COLOR);
		if(image.data == NULL) continue;


		if (!image.empty()) {

			if(trackObject < 0) {
				tracker.newTracker(image, selection);
				trackObject = 1;
			}

			if (trackObject) {
				tracker.updateTracker(image);
				//ellipse( image, tracker.track(image), Scalar(0,0,255), 3, CV_AA );
				//tracker.track(image);
				//drawRectangle(&image, tracker.getTrackWindow());
				//drawRectangle(&image, selection);
				//imshow("projection", tracker.getHistogramProjection());
			}

			if (selectObject && selection.width > 0 && selection.height > 0) {
				Mat roi(image, selection);
				bitwise_not(roi, roi);
			}

			putText(image, "Hybrid Tracker", Point(20, 20),
					FONT_HERSHEY_SIMPLEX, 0.5f, Scalar(255, 255, 255));

			sprintf(img_file, "out/%04d.png", i);
			imwrite(img_file, image);
			imshow("Win", image);

			waitKey(30);
		} else
			i = 0;
	}

return 0;

}

