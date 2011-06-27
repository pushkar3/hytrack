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
	vector<vector<DMatch> > matches;
	vector<Mat> desc_vector;

	Mat roi_desc;
	int roi_keypoints;
	Rect trackWindow;

public:
	FeatureTracker() {
		detector = new SiftFeatureDetector(SIFT::DetectorParams::GET_DEFAULT_THRESHOLD(),
				SIFT::DetectorParams::GET_DEFAULT_EDGE_THRESHOLD(),
				SIFT::CommonParams::AVERAGE_ANGLE);
		// detector = new SurfFeatureDetector( 400, 3, 4);
		// hessian threshold, octaves, octave layers

		descriptor = new SurfDescriptorExtractor(3, 4, false);
		// octaves, octave_layers, extended

		matcher = new BruteForceMatcher<L2<float> > ();
		// SSD matcher

		roi_keypoints = 0;
	};

	~FeatureTracker() { };

	void init(Mat image, Rect selection) {
		vector<KeyPoint> keypoints;

		trackWindow = selection;
		Mat roi(image, selection);
		desc_vector.clear();
		detector->detect(roi, keypoints);

		if (keypoints.size() > 0) {
			roi_keypoints = keypoints.size();
			descriptor->compute(roi, keypoints, roi_desc);

			for (int i = 0; i < keypoints.size(); i++) {
				int x = selection.x + keypoints[i].pt.x;
				int y = selection.y + keypoints[i].pt.y;
				circle(image, Point(x, y), 2, CV_RGB(0, 255, 0));
			}
		}

	}

	Rect track(Mat image) {
		vector<KeyPoint> keypoints;
		Mat desc;

		detector->detect(image, keypoints);
		if(keypoints.size() > 0) {
			descriptor->compute(image, keypoints, desc);
			for (int i = 0; i < keypoints.size(); i++)
				circle(image, keypoints[i].pt, 2, CV_RGB(0, 255, 0));

			matcher->clear();
			matcher->knnMatch(desc, roi_desc, matches, roi_keypoints);
			cout << "Matches size is " << matches.size() << endl;
		}

		return trackWindow;
	}
};

#endif /* FEATURETRACKER_H_ */
