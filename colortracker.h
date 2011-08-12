#ifndef HYBRIDTRACKER_H_
#define HYBRIDTRACKER_H_

#include <cvaux.h>
#include <iostream>

using namespace cv;
using namespace std;

class HybridTracker {
public:
	Mat hsv;
	Mat backproj;
	Mat mask, maskroi;
	MatND hist;
	Rect trackwindow;

	CvMat* samples;
	CvMat* labels;
	CvEM em_model;
	CvEMParams params;

	HybridTracker() { 	}

	~HybridTracker() {	}

	void init(Mat image, Rect selection) {

		int histSize[] = { image.rows, image.cols };
		int channels[] = { 0, 1, 2 };
		float hrange[] = { 0, 180 };
		float srange[] = { 0, 256 };
		float vrange[] = { 0, 1 };
		const float* ranges[] = { hrange, srange, vrange };

		cvtColor(image, hsv, CV_BGR2HSV);
		inRange(hsv, Scalar(0, 0, 0), Scalar(256, 256, 256), mask);

		Mat hsv_roi(hsv, selection);
		Mat maskroi(mask, selection);
		calcHist(&hsv_roi, 1, channels, maskroi, hist, 2, histSize, ranges, true, false);
		normalize(hist, hist, 0, 255, CV_MINMAX);
		calcBackProject(&hsv, 1, channels, hist, backproj, ranges);

		trackwindow = selection;
	}


	void track(Mat image) {
		int channels[] = { 0, 1, 2 };
		float hrange[] = { 0, 180 };
		float srange[] = { 0, 256 };
		float vrange[] = { 0, 1 };
		const float* ranges[] = { hrange, srange, vrange };

		cvtColor(image, hsv, CV_BGR2HSV);
		calcBackProject(&hsv, 1, channels, hist, backproj, ranges);
		meanShift(backproj, trackwindow, TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1));
	}

	void setTrackWindow(Rect _window) {
		trackwindow = _window;
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

#endif /* HYBRIDTRACKER_H_ */
