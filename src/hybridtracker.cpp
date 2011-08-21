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
#include <stdio.h>
#include <iostream>
#include <highgui.h>
#include "precomp.hpp"
#include "opencv2/tracker/hybridtracker.hpp"

using namespace cv;
using namespace std;

CvHybridTracker::CvHybridTracker()
{

}


CvHybridTracker::CvHybridTracker(HybridTrackerParams params)
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

void CvHybridTracker::set(Mat image, Rect selection)
{
	_size = image.size();
	w_ms = 0.5;
	w_ft = 0.5;
	mstracker->init(image, selection);
	fttracker->init(image, selection);

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

	samples = cvCreateMat(2, 2, CV_32FC1);
	labels = cvCreateMat(2, 1, CV_32SC1);
}

float CvHybridTracker::getL2Norm(Point2d p1, Point2d p2)
{
	float distance = (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);
	return sqrt(distance);
}

Mat CvHybridTracker::getDistanceProjection(Point2d center)
{
	Mat hist(_size, CV_64F);

	double lu = getL2Norm(Point(0, 0), center);
	double ru = getL2Norm(Point(0, _size.width), center);
	double rd = getL2Norm(Point(_size.height, _size.width), center);
	double ld = getL2Norm(Point(_size.height, 0), center);

	double max = (lu < ru) ? lu : ru;
	max = (max < rd) ? max : rd;
	max = (max < ld) ? max : ld;

	for (int i = 0; i < hist.rows; i++)
		for (int j = 0; j < hist.cols; j++)
			hist.at<double> (i, j) = 1.0 - (getL2Norm(Point(i, j), center)/max);

	return hist;
}

Mat CvHybridTracker::getGaussianProjection(int ksize, double sigma,	Point2d center)
{
	Mat kernel = getGaussianKernel(ksize, sigma, CV_64F);
	double max = kernel.at<double> (ksize/2);

	Mat hist(_size, CV_64F);
	for (int i = 0; i < hist.rows; i++)
	{
		for (int j = 0; j < hist.cols; j++)
		{
			int pos = getL2Norm(Point(i, j), center);
			if (pos < ksize / 2.0)
				hist.at<double> (i, j) = 1.0 - (kernel.at<double> (pos)/max);
		}
	}


	return hist;
}

void CvHybridTracker::mergeTrackers(Mat image)
{
	mstracker->track(image);
	fttracker->track(image);
	Mat ms_backproj = mstracker->backproj;
	Mat ms_backproj_f(_size, CV_64F);
	ms_backproj.convertTo(ms_backproj_f, CV_64F);
	Mat ms_distproj = getDistanceProjection(mstracker->center);
	Mat ms_proj = ms_backproj_f.mul(ms_distproj);

	float dist_err = getL2Norm(mstracker->center, fttracker->center);
	Mat ft_gaussproj = getGaussianProjection(dist_err, -1, fttracker->center);
	Mat ft_distproj = getDistanceProjection(fttracker->center);
	Mat ft_proj = ft_gaussproj.mul(ft_distproj);

	Mat proj = w_ms * ms_proj + w_ft * ft_proj;

//	imshow("ms_proj", ms_proj);
//	imshow("ms_distproj", ms_distproj);
//	imshow("ft_gaussproj", ft_gaussproj);
	imshow("ft_distproj", ft_distproj);

//	samples->data.fl[0] = mstracker->center.x;
//	samples->data.fl[1] = mstracker->center.y;
//	samples->data.fl[2] = fttracker->center.x;
//	samples->data.fl[3] = fttracker->center.y;
//
//	em_model.train(samples, 0, params, labels);
//	center = em_model.getMeans().at<Point2d> (0);
}
