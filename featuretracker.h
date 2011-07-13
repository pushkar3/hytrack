#ifndef FEATURETRACKER_H_
#define FEATURETRACKER_H_

#include <cv.h>
#include <cvaux.h>
#include <iostream>

using namespace cv;
using namespace std;

class FeatureTracker {
private:
	FeatureDetector* detector;
	DescriptorExtractor* descriptor;
	DescriptorMatcher* matcher;
	vector<DMatch> matches;
	vector<Mat> desc_vector;

	Mat roi_desc;
	vector<KeyPoint> roi_keypoints;
	Point roi_origin;
	int roi_keys;
	Rect trackWindow;
	int window_searchsize;

public:
	FeatureTracker() {
		detector = new SiftFeatureDetector(SIFT::DetectorParams::GET_DEFAULT_THRESHOLD(),
				SIFT::DetectorParams::GET_DEFAULT_EDGE_THRESHOLD(),
				SIFT::CommonParams::AVERAGE_ANGLE);
		// detector = new SurfFeatureDetector( 400, 3, 4);
		// hessian threshold, octaves, octave layers

		descriptor = new SurfDescriptorExtractor(3, 4, false);
		// octaves, octave_layers, extended

		matcher = new  BruteForceMatcher<L2<float> > ();
		// SSD matcher

		roi_keys = 0;
		window_searchsize = 0;
	};

	~FeatureTracker() { };

	// Calculate keypoints of the roi
	// Store keypoints and center of the roi
	void init(Mat image, Rect selection) {
		trackWindow = selection;
		Mat roi(image, selection);
		desc_vector.clear();
		detector->detect(roi, roi_keypoints);

		if (roi_keypoints.size() > 0) {
			roi_keys = roi_keypoints.size();
			descriptor->compute(roi, roi_keypoints, roi_desc);

			for (int i = 0; i < roi_keypoints.size(); i++) {
				roi_keypoints[i].pt.x += selection.x;
				roi_keypoints[i].pt.y += selection.y;
				circle(image, roi_keypoints[i].pt, 2, CV_RGB(0, 255, 0));
			}

			roi_origin.x = selection.x + selection.width/2.0;
			roi_origin.y = selection.y + selection.height/2.0;
		}

	}

	void setTrackWindow(Rect _window) {
		trackWindow = _window;
	}

	// Calculate keypoints around the roi
	// Match keypoints and try to find new center of the roi
	// TODO: Very unstable!
	Rect track(Mat image) {
		vector<KeyPoint> keypoints;
		Mat desc;
		Rect newtrackWindow = trackWindow;

		newtrackWindow.x -= window_searchsize;
		newtrackWindow.y -= window_searchsize;
		newtrackWindow.width += window_searchsize;
		newtrackWindow.height += window_searchsize;
		if(newtrackWindow.x < 0) newtrackWindow.x = 0;
		if(newtrackWindow.y < 0) newtrackWindow.y = 0;

		Mat roi(image, newtrackWindow);
		detector->detect(roi, keypoints);
		if(keypoints.size() > 10) {
			descriptor->compute(roi, keypoints, desc);
			for (int i = 0; i < keypoints.size(); i++) {
				keypoints[i].pt.x += newtrackWindow.x;
				keypoints[i].pt.y += newtrackWindow.y;
				circle(image, keypoints[i].pt, 2, CV_RGB(0, 255, 0));
			}

			matcher->match(roi_desc, desc, matches);


			Point origin;
			for(int i = 0; i < matches.size(); i++) {
				//line(image, roi_keypoints[matches[i].queryIdx].pt, keypoints[matches[i].trainIdx].pt, Scalar(0, 0, 255), 1, CV_AA, 0);
				origin.x += keypoints[matches[i].trainIdx].pt.x;
				origin.y += keypoints[matches[i].trainIdx].pt.y;
			}

			origin.x /= matches.size();
			origin.y /= matches.size();

			trackWindow.x += (origin.x - roi_origin.x);
			trackWindow.y += (origin.y - roi_origin.y);
			roi_origin = origin;
		}

		return trackWindow;
	}
};

#endif /* FEATURETRACKER_H_ */
