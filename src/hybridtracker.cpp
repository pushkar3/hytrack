//*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                                License Agreement
//                       For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2008-2011, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#include "precomp.hpp"
#include "opencv2/tracker/hybridtracker.hpp"

using namespace cv;
using namespace std;

CvHybridTracker::CvHybridTracker()
{

}


CvHybridTracker::CvHybridTracker(HybridTrackerParams _params) : params(_params)
{
	params.ft_params.feature_type = CvFeatureTrackerParams::SIFT;
	mstracker = new CvMeanShiftTracker(params.ms_params);
	fttracker = new CvFeatureTracker(params.ft_params);
}

CvHybridTracker::~CvHybridTracker()
{
	if(mstracker != NULL) delete mstracker;
	if(fttracker != NULL) delete fttracker;
}

inline float CvHybridTracker::getL2Norm(Point2d p1, Point2d p2)
{
	float distance = (p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y);
	return sqrt(distance);
}

Mat CvHybridTracker::getDistanceProjection(Mat image, Point2d center)
{
	Mat hist(image.size(), CV_64F);

	double lu = getL2Norm(Point(0, 0), center);
	double ru = getL2Norm(Point(0, image.size().width), center);
	double rd = getL2Norm(Point(image.size().height, image.size().width), center);
	double ld = getL2Norm(Point(image.size().height, 0), center);

	double max = (lu < ru) ? lu : ru;
	max = (max < rd) ? max : rd;
	max = (max < ld) ? max : ld;

	for (int i = 0; i < hist.rows; i++)
		for (int j = 0; j < hist.cols; j++)
			hist.at<double> (i, j) = 1.0 - (getL2Norm(Point(i, j), center)/max);

	return hist;
}

Mat CvHybridTracker::getGaussianProjection(Mat image, int ksize, double sigma,	Point2d center)
{
	Mat kernel = getGaussianKernel(ksize, sigma, CV_64F);
	double max = kernel.at<double> (ksize/2);

	Mat hist(image.size(), CV_64F);
	for (int i = 0; i < hist.rows; i++)
		for (int j = 0; j < hist.cols; j++)
		{
			int pos = getL2Norm(Point(i, j), center);
			if (pos < ksize / 2.0)
				hist.at<double> (i, j) = 1.0 - (kernel.at<double> (pos)/max);
		}

	return hist;
}


void CvHybridTracker::newTracker(Mat image, Rect selection)
{
	prev_proj = Mat::zeros(image.size(), CV_64FC1);
	prev_center = Point2f(selection.x+selection.width/2.0, selection.y+selection.height/2.0);
	prev_window = selection;

	mstracker->newTrackingWindow(image, selection);
	fttracker->newTrackingWindow(image, selection);

	params.em_params.covs = NULL;
	params.em_params.means = NULL;
	params.em_params.probs = NULL;
	params.em_params.nclusters = 1;
	params.em_params.weights = NULL;
	params.em_params.cov_mat_type = CvEM::COV_MAT_DIAGONAL;
	params.em_params.start_step = CvEM::START_AUTO_STEP;
	params.em_params.term_crit.max_iter = 10;
	params.em_params.term_crit.epsilon = 0.1;
	params.em_params.term_crit.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;

	samples = cvCreateMat(2, 10, CV_32FC1);
	labels = cvCreateMat(2, 1, CV_32SC1);
}

void CvHybridTracker::updateTracker(Mat image)
{
	mstracker->updateTrackingWindow(image);
	fttracker->updateTrackingWindow(image);

//	if (params.motion_model = CvMotionModel::EM)
//		updateTrackerWithEM(image);
//	else
	updateTrackerWithLowPassFilter(image);

	mstracker->setTrackingWindow(prev_window);
	fttracker->setTrackingWindow(prev_window);
}

void CvHybridTracker::updateTrackerWithEM(Mat image)
{
	Mat ms_backproj = mstracker->getHistogramProjection(CV_64F);
	Mat ms_distproj = getDistanceProjection(image, mstracker->getTrackingCenter());
	Mat ms_proj = ms_backproj.mul(ms_distproj);

	float dist_err = getL2Norm(mstracker->getTrackingCenter(), fttracker->getTrackingCenter());
	Mat ft_gaussproj = getGaussianProjection(image, dist_err, -1, fttracker->getTrackingCenter());
	Mat ft_distproj = getDistanceProjection(image, fttracker->getTrackingCenter());
	Mat ft_proj = ft_gaussproj.mul(ft_distproj);

	Mat proj = params.ms_tracker_weight * ms_proj + params.ft_tracker_weight * ft_proj + prev_proj;
	int sample_count = countNonZero(proj);
	if(samples != NULL) cvReleaseMat(&samples);
	samples = cvCreateMat(2, sample_count, CV_32FC1);

	int count = 0;
	for (int i = 0; i < proj.rows; i++)
		for (int j = 0; j < proj.cols; j++)
			if (proj.at<double> (i, j) > 0)
			{
				samples->data.fl[count * 2] = i;
				samples->data.fl[count * 2 + 1] = j;
				count++;
			}

	params.em_params.means = em_model.get_means();
	params.em_params.covs = (const CvMat**) em_model.get_covs();
	params.em_params.weights = em_model.get_weights();

	em_model.train(samples, 0, params.em_params, labels);
	Point2f center = em_model.getMeans().at<Point2d> (0);
	prev_proj = proj;
	prev_center = center;
	prev_window.x = center.x;
	prev_window.y = center.y;
}

void CvHybridTracker::updateTrackerWithLowPassFilter(Mat image)
{
	RotatedRect ms_track = mstracker->getTrackingEllipse();
	Point2f ft_center = fttracker->getTrackingCenter();

	trackbox = ms_track;
	trackbox.center.x *= params.ms_tracker_weight*trackbox.center.x + params.ft_tracker_weight*ft_center.x;
	trackbox.center.y *= params.ms_tracker_weight*trackbox.center.y + params.ft_tracker_weight*ft_center.y;

	float a = params.low_pass_gain;
	trackbox.center.x *= a*trackbox.center.x + (1.0-a)*prev_center.x;
	trackbox.center.y *= a*trackbox.center.y + (1.0-a)*prev_center.y;

	prev_window.x = trackbox.center.x - prev_window.width/2.0;
	prev_window.y = trackbox.center.y - prev_window.height/2.0;
}

Rect CvHybridTracker::getTrackingWindow()
{
	return prev_window;
}

