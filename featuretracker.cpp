#include "hybridtracker.hpp"
#include <highgui.h>

CvFeatureTracker::CvFeatureTracker()
{
	detector = new SiftFeatureDetector(
			SIFT::DetectorParams::GET_DEFAULT_THRESHOLD(),
			SIFT::DetectorParams::GET_DEFAULT_EDGE_THRESHOLD(),
			SIFT::CommonParams::AVERAGE_ANGLE);
	// detector = new SurfFeatureDetector(400, 3, 4);


	descriptor = new SurfDescriptorExtractor(3, 4, false);

	matcher = new BruteForceMatcher<L2<float> > ();
	// SSD matcher
}


CvFeatureTracker::~CvFeatureTracker()
{
}

void CvFeatureTracker::init(Mat image, Rect selection)
{
	trackWindow = selection;
	prev_image = image;

	Mat mask = Mat::zeros(image.size(), CV_8UC1);
	rectangle(mask, Point(trackWindow.x, trackWindow.y), Point(trackWindow.x
			+ trackWindow.width, trackWindow.y + trackWindow.height), Scalar(
			255), CV_FILLED);

	prev_desc_vector.clear();
	detector->detect(prev_image, prev_keypoints, mask);

	if (prev_keypoints.size() > 0)
	{
		descriptor->compute(prev_image, prev_keypoints, prev_desc);
	}
}

void CvFeatureTracker::setTrackWindow(Rect _window)
{
	trackWindow = _window;
}

Rect CvFeatureTracker::track(Mat image)
{
	init(prev_image, trackWindow);

	vector<KeyPoint> current_keypoints;
	Mat current_desc;

	int windowSize = 10;
	Rect window(trackWindow.x - windowSize, trackWindow.y - windowSize,
			trackWindow.width + windowSize, trackWindow.height + windowSize);

	Mat mask = Mat::zeros(image.size(), CV_8UC1);
	rectangle(mask, Point(window.x, window.y), Point(window.x + window.width,
			window.y + window.height), Scalar(255), CV_FILLED);

	detector->detect(image, current_keypoints, mask);

	if (current_keypoints.size() > 4)
	{
		descriptor->compute(image, current_keypoints, current_desc);
		matcher->match(prev_desc, current_desc, matches);

		Point p0 = prev_keypoints[matches[0].trainIdx].pt;
		Point n0 = current_keypoints[matches[0].queryIdx].pt;

#if 0
		Point p1 = prev_keypoints[matches[1].trainIdx].pt;
		Point n1 = current_keypoints[matches[1].queryIdx].pt;

		double dp = sqrt((p0.x - p1.x)*(p0.x - p1.x) + (p0.y - p1.y)*(p0.y - p1.y));
		double dn = sqrt((n0.x - n1.x)*(n0.x - n1.x) + (n0.y - n1.y)*(n0.y - n1.y));
		printf("scale: %lf, %lf\n", dn, dp);
		double scale = dn/dp;
		trackWindow.width *= scale;
		trackWindow.height *= scale;
#endif


		trackWindow.x += (p0.x - n0.x);
		trackWindow.y += (p0.y - n0.y);

		rectangle(image, Point(trackWindow.x, trackWindow.y), Point(trackWindow.x + trackWindow.width,
					trackWindow.y + trackWindow.height), Scalar(255));
		Mat disp;
		drawMatches(prev_image, prev_keypoints, image, current_keypoints, matches, disp);
		imshow("disp", disp);
	}

	return trackWindow;
}

