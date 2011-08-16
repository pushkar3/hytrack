#ifndef MSTRACKER_H_
#define MSTRACKER_H_

#include <cvaux.h>
#include <iostream>

using namespace cv;
using namespace std;

class MeanShiftTracker {
public:
	Mat hsv, hue;
	Mat backproj;
	Mat mask, maskroi;
	MatND hist;
	Rect trackwindow;

	CvMat* samples;
	CvMat* labels;
	CvEM em_model;
	CvEMParams params;

	MeanShiftTracker() { 	}

	~MeanShiftTracker() {	}

	void init(Mat image, Rect selection) {

		hist.release();
		int histSize = 16;
		int channels[] = { 0, 0 };
		float hrange[] = { 0, 180 };
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


	RotatedRect track(Mat image) {
		int channels[] = { 0, 0 };
		float hrange[] = { 0, 180 };
		const float* ranges = hrange;

		cvtColor(image, hsv, CV_BGR2HSV);
		inRange(hsv, Scalar(0, 30, MIN(10, 256)), Scalar(180, 256, MAX(10, 256)), mask);
		mixChannels(&hsv, 1, &hue, 1, channels, 1);
		calcBackProject(&hue, 1, 0, hist, backproj, &ranges);
		//normalize(backproj, backproj, 0, 255, CV_MINMAX);
		backproj &= mask;
		// meanShift(backproj, trackwindow, TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1));
		RotatedRect trackbox = CamShift(backproj, trackwindow, TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));
		int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5)/6;
		trackwindow = Rect(trackwindow.x - r, trackwindow.y - r, trackwindow.x + r, trackwindow.y + r) & Rect(0, 0, cols, rows);
		return trackbox;
	}

	void fillForEM() {
		int sample_cnt = 0;
		for(int j = 0; j < mask.rows; j++) {
			for(int k = 0; k < mask.cols; k++) {
				if(mask.at<unsigned char>(j, k) != 0) {
					sample_cnt++;
				}
			}
		}
	}

	void initEM() {
		int sample_cnt = 10;
		samples = cvCreateMat(sample_cnt, 2, CV_32FC1);
		labels = cvCreateMat(sample_cnt, 1, CV_32SC1);

		params.covs = NULL;
		params.means = NULL;
		params.weights = NULL;
		params.probs = NULL;
		params.nclusters = 1;
		params.cov_mat_type = CvEM::COV_MAT_DIAGONAL;
		params.start_step = CvEM::START_AUTO_STEP;
		params.term_crit.max_iter = 10;
		params.term_crit.epsilon = 0.1;
		params.term_crit.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;
	}

	void runEM() {
		em_model.train(samples, 0, params, labels);
		for (int i = 0; i < em_model.get_nclusters(); i++) {
			cout << "Mean " << i << " is " << em_model.getMeans().at<double> (
					i, 0) << ", " << em_model.getMeans().at<double> (i, 1)
					<< endl;
//			circle(mask, Point(em_model.getMeans().at<double> (i, 0),
//					em_model.getMeans().at<double> (i, 1)), 4, 127, -1);
		}
	}

	void closeEM() {
		cvReleaseMat(&samples);
		cvReleaseMat(&labels);
	}
};

#endif
