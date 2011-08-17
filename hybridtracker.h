#ifndef HYBRIDTRACKER_H_
#define HYBRIDTRACKER_H_

#include <cv.h>
#include <cvaux.h>
#include <iostream>

#include "colortracker.h"
#include "featuretracker.h"

using namespace cv;
using namespace std;

class HybridTracker {
public:
	enum tracker_t { TRACKER_MEANSHIFT, TRACKER_SIFT , TRACKER_SIFTRANSAC , TOTAL_TRACKERS};
	enum kernel_t { KERNEL_DISTANCE, KERNEL_NONE };
	float w[TOTAL_TRACKERS];
	Mat backproj[TOTAL_TRACKERS];

	Size _size;

	MeanShiftTracker mstracker;
	FeatureTracker fttracker;

	CvMat* samples;
	CvMat* labels;
	CvEM em_model;
	CvEMParams params;

	Point2d center;

	int w_ms, w_ft;

public:
	HybridTracker() { }
	~HybridTracker() { }

	void set(Mat image, Rect selection) {
		_size = image.size();
		w_ms = 0.5;
		w_ft = 0.5;
		mstracker.init(image, selection);
		fttracker.init(image, selection);

		params.covs = NULL;
		params.means = NULL;
		params.probs = NULL;
		params.nclusters = 1;
		params.weights = NULL;
		params.cov_mat_type = CvEM::COV_MAT_DIAGONAL;
		params.start_step = CvEM::START_AUTO_STEP;
		params.term_crit.max_iter = 10;
		params.term_crit.epsilon = 0.1;
		params.term_crit.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;

		samples = cvCreateMat(_size.width*_size.height, 2, CV_32FC1);
		labels = cvCreateMat(_size.width*_size.height, 1, CV_32SC1);
	}

	float getL2Norm(Point2d p1, Point2d p2) {
		float distance = (p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y);
		return sqrt(distance);
	}

	Mat getDistanceProjection(Point2d center) {
		Mat hist(_size, CV_64F);

		float lu = getL2Norm(Point(0, 0), center);
		float ru = getL2Norm(Point(0, _size.width), center);
		float rd = getL2Norm(Point(_size.height, _size.width), center);
		float ld = getL2Norm(Point(_size.height, 0), center);

		float max = (lu < ru) ? lu : ru;
		max = (max < rd) ? max : rd;
		max = (max < ld) ? max : ld;

		for(int i = 0; i < hist.rows; i++) {
			for (int j = 0; j < hist.cols; j++) {
				hist.at<double>(i, j) = max - getL2Norm(Point(i, j), center);
			}
		}

		normalize(hist, hist, 255, 0, NORM_L2);

		return hist;
	}

	Mat getGaussianProjection(int ksize, double sigma, Point2d center) {
		Mat kernel = getGaussianKernel(ksize, sigma, CV_64F);
		kernel *= (255.0f/kernel.at<double>(ksize/2, 1));

		Mat hist(_size, CV_64F);
		for (int i = 0; i < hist.rows; i++) {
			for (int j = 0; j < hist.cols; j++) {
				int pos = getL2Norm(Point(i, j), center);
				if(pos < ksize/2.0)
					hist.at<double> (i, j) = 255-kernel.at<double> (pos);
			}
		}
		normalize(hist, hist, 255, 0, NORM_L2);
		return hist;
	}

	void mergeTrackers(Mat image) {
		mstracker.track(image);
		fttracker.track(image);
		Mat ms_backproj = mstracker.backproj;
		Mat ms_backproj_f(_size, CV_64F);
		ms_backproj.convertTo(ms_backproj_f, CV_64F);
		Mat ms_distproj = getDistanceProjection(mstracker.center);
		Mat ms_proj = ms_backproj_f.mul(ms_distproj);

		float dist_err = getL2Norm(mstracker.center, fttracker.center);
		Mat ft_gaussproj = getGaussianProjection(dist_err, -1, fttracker.center);
		Mat ft_distproj = getDistanceProjection(fttracker.center);
		Mat ft_proj = ft_gaussproj.mul(ft_distproj);

		Mat proj = w_ms * ms_proj + w_ft * ft_proj;

		int cnt = 0;
		for (int i = 0; i < _size.height; i++) {
			for (int j = 0; j < _size.width; j++) {
				samples->data.fl[cnt*2] = i;
				samples->data.fl[cnt*2+1] = j;
				cnt++;
			}
		}

		em_model.train(samples, 0, params, labels);
		center = em_model.getMeans().at<Point2d>(0);

		for (int i = 0; i < em_model.get_nclusters(); i++)
			cout << "Mean " << i << " is " << em_model.getMeans().at<double> (i, 0) << ", " << em_model.getMeans().at<double> (i, 1) << endl;
	}
};


#endif /* HYBRIDTRACKER_H_ */

