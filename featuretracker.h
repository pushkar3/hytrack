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
	vector<KeyPoint> keypoints;
	DescriptorMatcher* matcher;
	vector<DMatch> matches;
	vector<Mat> desc_vector;

	Mat desc;
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
	};

	~FeatureTracker() { };

	void init(Mat image, Rect selection) {
		trackWindow = selection;
		Mat roi(image, selection);
		desc_vector.clear();
		detector->detect(roi, keypoints);

		if(keypoints.size() > 0) {
			descriptor->compute(image, keypoints, desc);
			desc_vector.push_back(desc);
		}

		for(int i = 0; i < keypoints.size(); i++) {
			int x = selection.x + keypoints[i].pt.x;
			int y = selection.y + keypoints[i].pt.y;
			circle(image, Point(x, y), 2, CV_RGB(0, 255, 0));
		}
	}

	Rect track(Mat image) {
		return trackWindow;
	}

	void match() {
		matcher = new BruteForceMatcher<L2<float> > ();

		matcher->clear();
		matcher->add(desc_vector);
		matcher->match(desc, matches);
	}
};

#endif /* FEATURETRACKER_H_ */
