#ifndef COLORTRACKER_H_
#define COLORTRACKER_H_

#include <cvaux.h>
#include <iostream>

using namespace cv;
using namespace std;

class ColorTracker {
private:
	Mat hsv, mask, hue, hist, backproj;
	Rect trackWindow;
	int hsize;

public:
	ColorTracker() {
		hsize = 16;
	}

	~ColorTracker() { }

	void init(Mat image, Rect selection /* Rect mask */) {
		float hranges[] = {0,180};
		const float* phranges = hranges;
		cvtColor(image, hsv, CV_BGR2HSV);
		inRange(hsv, Scalar(0, 30, MIN(10, 256)), Scalar(180, 256, MAX(10, 256)), mask);
		int ch[] = {0, 0};
		hue.create(hsv.size(), hsv.depth());
		Mat roi(hue, selection), maskroi(mask, selection);
		calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
		normalize(hist, hist, 0, 255, CV_MINMAX);
		trackWindow = selection;
	}

	void setTrackWindow(Rect _window) {
		trackWindow = _window;
	}

	Rect track(Mat image) {
		float hranges[] = {0,180};
		const float* phranges = hranges;
		cvtColor(image, hsv, CV_BGR2HSV);
		inRange(hsv, Scalar(0, 30, MIN(10, 256)), Scalar(180, 256, MAX(10, 256)), mask);
		int ch[] = {0, 0};
		hue.create(hsv.size(), hsv.depth());
		mixChannels(&hsv, 1, &hue, 1, ch, 1);
		calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
		backproj &= mask;
		meanShift(backproj, trackWindow, TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));
		return trackWindow;
	}
};

#endif /* COLORTRACKER_H_ */
