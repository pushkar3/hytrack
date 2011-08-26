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

Mat frame, image;
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



int main(int argc, char** argv)
{
	VideoCapture cap;

	cap.open(0);
	if (!cap.isOpened())
	{
		cout << "Failed to open camera" << endl;
		return 0;
	}
	cout << "Opened camera" << endl;
	cap >> frame;

	HybridTrackerParams params;
	// motion model params
	params.motion_model = CvMotionModel::LOW_PASS_FILTER;
	params.low_pass_gain = 0.1;
	// mean shift params
	params.ms_tracker_weight = 0.8;
	params.ms_params.tracking_type = CvMeanShiftTrackerParams::HS;
	// feature tracking params
	params.ft_tracker_weight = 0.2;
	params.ft_params.feature_type = CvFeatureTrackerParams::SIFT;
	params.ft_params.window_size = 0;

	HybridTracker tracker(params);
	char img_file[20] = "seqG/0001.png";
	char img_file_num[10];
	namedWindow("Win", 1);

	setMouseCallback("Win", onMouse, 0);

	for (int i = 0; i < 1000; i++)
	{
//		sprintf(img_file, "seqG/%04d.png", i);
//		image = imread(img_file, CV_LOAD_IMAGE_COLOR);

		cap >> frame;
		frame.copyTo(image);
		if (image.data == NULL)
			continue;

		sprintf(img_file_num, "Frame: %d", i);
		putText(image, img_file_num, Point(10, image.rows-20), FONT_HERSHEY_PLAIN, 0.75, Scalar(255, 255, 255));
		if (!image.empty())
		{

			if (trackObject < 0)
			{
				tracker.newTracker(image, selection);
				trackObject = 1;
			}

			if (trackObject)
			{
				tracker.updateTracker(image);
				drawRectangle(&image, tracker.getTrackingWindow());

			}

			if (selectObject && selection.width > 0 && selection.height > 0)
			{
				Mat roi(image, selection);
				bitwise_not(roi, roi);
			}

			imshow("Win", image);

			waitKey(100);
		}
		else
			i = 0;
	}

	return 0;

}
