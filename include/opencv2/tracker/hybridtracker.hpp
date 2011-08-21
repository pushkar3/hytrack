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

#ifndef __OPENCV_HYBRIDTRACKER_H_
#define __OPENCV_HYBRIDTRACKER_H_

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/ml/ml.hpp"

#ifdef __cplusplus

struct CvMeanShiftTrackerParams
{
	CvMeanShiftTrackerParams() { };
	CvTermCriteria term_crit;
};

struct CvFeatureTrackerParams
{
	enum { SIFT = 0, SURF = 1 };
	CvFeatureTrackerParams(int feature_type = 0, int window_size = 0)
	{
		feature_type = 0;
		window_size = 0;
	}

	int feature_type;
	int window_size;
};

struct CvHybridTrackerParams
{
	CvHybridTrackerParams() { };

	CvFeatureTrackerParams ft_params;
	CvMeanShiftTrackerParams ms_params;
	CvEMParams em_params;
};

namespace cv
{

class CvMeanShiftTracker
{
	CvMeanShiftTrackerParams params;
public:
	Mat hsv, hue;
	Mat backproj;
	Mat mask, maskroi;
	MatND hist;
	Rect trackwindow;
	RotatedRect trackbox;

	Point2d center;

	CvMeanShiftTracker();
	CvMeanShiftTracker(CvMeanShiftTrackerParams _params = CvMeanShiftTrackerParams());
	~CvMeanShiftTracker();
	void init(Mat image, Rect selection);
	RotatedRect track(Mat image);
	void setTrackWindow(Rect window);
	Rect getTrackWindow();
	Mat getHistogramProjection();
};

class CvFeatureTracker
{
private:
	FeatureDetector* detector;
	DescriptorExtractor* descriptor;
	DescriptorMatcher* matcher;
	vector<DMatch> matches;

	Mat prev_image;
	vector<Mat> prev_desc_vector;
	vector<KeyPoint> prev_keypoints;
	Mat prev_desc;

	Rect trackWindow;

public:

	Point2d center;

	CvFeatureTracker();
	CvFeatureTracker(CvFeatureTrackerParams params = CvFeatureTrackerParams(0, 0));
	~CvFeatureTracker();
	void init(Mat image, Rect selection);
	void setTrackWindow(Rect _window);
	Rect track(Mat image);
};

class CvHybridTracker
{
public:
	Size _size;

	CvMeanShiftTracker* mstracker;
	CvFeatureTracker* fttracker;

	CvMat* samples;
	CvMat* labels;
	CvEM em_model;
	CvEMParams params;

	Point2d center;

	int w_ms, w_ft;

public:
	CvHybridTracker();
	CvHybridTracker(CvHybridTrackerParams params = CvHybridTrackerParams());
	~CvHybridTracker();
	void set(Mat image, Rect selection);
	float getL2Norm(Point2d p1, Point2d p2);
	Mat getDistanceProjection(Point2d center);
	Mat getGaussianProjection(int ksize, double sigma, Point2d center);
	void mergeTrackers(Mat image);
};

typedef CvMeanShiftTrackerParams MeanShiftTrackerParams;
typedef CvFeatureTrackerParams FeatureTrackerParams;
typedef CvHybridTrackerParams HybridTrackerParams;
typedef CvMeanShiftTracker MeanShiftTracker;
typedef CvFeatureTracker FeatureTracker;
typedef CvHybridTracker HybridTracker;
}

#endif

#endif
