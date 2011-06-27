#ifndef FEATURETRACKER_H_
#define FEATURETRACKER_H_

#include <cv.h>
#include <cvaux.h>
#include <iostream>

using namespace cv;
using namespace std;

class FeatureTracker {
private:
	Mat img;
	FeatureDetector* detector;
	DescriptorExtractor* descriptor;
	vector<KeyPoint> keypoints;
	DescriptorMatcher* matcher;
	vector<DMatch> matches;
	vector<Mat> desc_vector;

	Mat desc;
	Rect roi;

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

	void priorImage(Mat _img) {
		img = _img.clone();
	}

	void findDescriptors() {
		detector->detect(img, keypoints);
		cout << "Number of keypoints: " << keypoints.size() << endl;

		if(keypoints.size() > 0) {
			descriptor->compute(img, keypoints, desc);
			desc_vector.push_back(desc);
		}

		for(int i = 0; i < keypoints.size(); i++) {
			circle(img, keypoints[i].pt, 2, CV_RGB(0, 255, 0));
		}

		int channels[] = {0};
		Mat hist;
		Mat backproj;
		float hranges[] = {0,180};
		const float* phranges = hranges;

 		calcBackProject(&desc, 1, channels, hist, backproj, &phranges);

 		meanShift(backproj, roi, TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1));
	}

	void match() {
		matcher = new BruteForceMatcher<L2<float> > ();

		matcher->clear();
		matcher->add(desc_vector);
		matcher->match(desc, matches);
	}

	void addROI(Rect _roi) {
		roi = _roi;
	}

	Mat updateTracks(Mat _img) {
		img = _img;
		rectangle(img, Point(roi.x, roi.y), Point(roi.x+roi.width, roi.y+roi.height), Scalar(255, 255, 255));

		int dtop = -roi.y;
		int dbottom = -img.rows+roi.y+roi.height;
		int dleft = -roi.x;
		int dright = -img.cols+roi.x+roi.width;

		img.adjustROI(dtop, dbottom, dleft, dright);
		findDescriptors();

		img.adjustROI(-dtop, -dbottom, -dleft, -dright);
		return img;
	}
};

#endif /* FEATURETRACKER_H_ */
