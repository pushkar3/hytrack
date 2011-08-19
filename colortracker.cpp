#include "hybridtracker.hpp"

CvMeanShiftTracker::CvMeanShiftTracker()
{
}

CvMeanShiftTracker::~CvMeanShiftTracker()
{
}

void CvMeanShiftTracker::init(Mat image, Rect selection)
{

	hist.release();
	int histSize = 16;
	int channels[] =
	{ 0, 0 };
	float hrange[] =
	{ 0, 180 };
	const float* ranges = hrange;

	cvtColor(image, hsv, CV_BGR2HSV);
	inRange(hsv, Scalar(0, 0, 0), Scalar(256, 256, 256), mask);

	hue.create(hsv.size(), hsv.depth());
	Mat roi(hue, selection);
	Mat maskroi(mask, selection);
	calcHist(&roi, 1, 0, maskroi, hist, 1, &histSize, &ranges);
	normalize(hist, hist, 0, 255, CV_MINMAX);

	trackwindow = selection;
}

RotatedRect CvMeanShiftTracker::track(Mat image)
{
	int channels[] =
	{ 0, 0 };
	float hrange[] =
	{ 0, 180 };
	const float* ranges = hrange;

	cvtColor(image, hsv, CV_BGR2HSV);
	inRange(hsv, Scalar(0, 30, MIN(10, 256)), Scalar(180, 256, MAX(10, 256)),
			mask);
	mixChannels(&hsv, 1, &hue, 1, channels, 1);
	calcBackProject(&hue, 1, 0, hist, backproj, &ranges);
	//normalize(backproj, backproj, 0, 255, CV_MINMAX);
	backproj &= mask;
	// meanShift(backproj, trackwindow, TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1));
	RotatedRect trackbox = CamShift(backproj, trackwindow, TermCriteria(
			CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1));
	int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5)
			/ 6;
	trackwindow = Rect(trackwindow.x - r, trackwindow.y - r, trackwindow.x + r,
			trackwindow.y + r) & Rect(0, 0, cols, rows);
	center.x = trackwindow.x + trackwindow.width / 2;
	center.y = trackwindow.y + trackwindow.height / 2;
	return trackbox;

}
