/*
 * cvTracker.h
 *
 *  Created on: Jun 5, 2011
 *      Author: pushkar
 */

#ifndef CVTRACKER_H_
#define CVTRACKER_H_

#include <cv.h>
#include <cvaux.h>
#include <iostream>

using namespace cv;
using namespace std;

class CvROITracker {
private:
	int w, h;
	CvSeq* x, y;		// Track of this object
	CvSeq* histogram;	// For Mean Shift
	CvSeq* features; 	// or use CvFeatureTree
						// SIFT features
	int w_ms;			// Param1 - Weight of shift based similarity measure
	int w_feature;		// Param2 - Weight of SOFT based similarity measure
public:
	CvROITracker(int x, int y, int w, int h);
	~CvROITracker();

	void update(CvMat* mat);
	// Uses some kernel to create a histogram
	// Creates a feature descriptor using some algorithms
	// Should the kernel and feature descriptor be specified by the user?
	// Or set optional params that will choose between various kernels and feature descriptors.

	void predict();
	void estimate();
	// Parameter estimation using EM

	void smooth();
	// Post processing of the path using Kalman filter

};

class CvTracker {
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
	CvTracker() {
		detector = new SiftFeatureDetector(SIFT::DetectorParams::GET_DEFAULT_THRESHOLD(),
				SIFT::DetectorParams::GET_DEFAULT_EDGE_THRESHOLD(),
				SIFT::CommonParams::AVERAGE_ANGLE);
		// detector = new SurfFeatureDetector( 400, 3, 4);
		// hessian threshold, octaves, octave layers

		descriptor = new SurfDescriptorExtractor(3, 4, false);
		// octaves, octave_layers, extended
	};

	~CvTracker() { };

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
		matcher = new BruteForceMatcher<L2<float> > ();

		matcher->clear();
		matcher->add(desc_vector);
		matcher->match(desc, matches);
		img.adjustROI(-dtop, -dbottom, -dleft, -dright);
		return img;
	}
};

#endif /* CVTRACKER_H_ */
