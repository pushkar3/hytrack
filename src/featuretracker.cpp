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
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/tracker/hybridtracker.hpp"

using namespace cv;

CvFeatureTracker::CvFeatureTracker(CvFeatureTrackerParams _params) : params(_params)
{
	switch (params.feature_type)
	{
	case CvFeatureTrackerParams::SIFT:
		detector = new SiftFeatureDetector(
				SIFT::DetectorParams::GET_DEFAULT_THRESHOLD(),
				SIFT::DetectorParams::GET_DEFAULT_EDGE_THRESHOLD(),
				SIFT::CommonParams::AVERAGE_ANGLE);
	case CvFeatureTrackerParams::SURF:
		detector = new SurfFeatureDetector(400, 3, 4);
	}

	descriptor = new SurfDescriptorExtractor(3, 4, false);

	matcher = new BruteForceMatcher<L2<float> > ();
}


CvFeatureTracker::~CvFeatureTracker()
{
}

void CvFeatureTracker::newTrackingWindow(Mat image, Rect selection)
{
	prev_image = image;
	prev_trackwindow = selection;
}

Rect CvFeatureTracker::updateTrackingWindow(Mat image)
{
	vector<KeyPoint> prev_keypoints, curr_keypoints;
	vector<Point2f> prev_keys, curr_keys;
	Mat prev_desc, curr_desc;

	Rect window = prev_trackwindow;
	Mat mask = Mat::zeros(image.size(), CV_8UC1);
	rectangle(mask, Point(window.x, window.y), Point(window.x + window.width, window.y + window.height), Scalar(255), CV_FILLED);

	detector->detect(prev_image, prev_keypoints, mask);

	window.x -= params.window_size;
	window.y -= params.window_size;
	window.width += params.window_size;
	window.height += params.window_size;
	rectangle(mask, Point(window.x, window.y), Point(window.x + window.width, window.y + window.height), Scalar(255), CV_FILLED);

	detector->detect(image, curr_keypoints, mask);

	if (prev_keypoints.size() > 8 && curr_keypoints.size() > 4)
	{
		descriptor->compute(prev_image, prev_keypoints, prev_desc);
		descriptor->compute(image, curr_keypoints, curr_desc);

		matcher->match(prev_desc, curr_desc, matches);

		for(int i = 0; i < matches.size(); i++)
		{
			prev_keys.push_back(prev_keypoints[matches[i].queryIdx].pt);
			curr_keys.push_back(curr_keypoints[matches[i].trainIdx].pt);
		}

		Mat T = findHomography(prev_keys, curr_keys, CV_RANSAC);

		prev_trackwindow.x += T.at<double>(0, 2);
		prev_trackwindow.y += T.at<double>(1, 2);

#if 0
		drawMatches(prev_image, prev_keypoints, image, curr_keypoints, matches, disp_matches);
		imshow("Matches", disp_matches);
		rectangle(image, Point(prev_trackwindow.x, prev_trackwindow.y), Point(prev_trackwindow.x + prev_trackwindow.width, prev_trackwindow.y + prev_trackwindow.height), Scalar(255, 255, 0), 2, CV_AA);
#endif
	}

	prev_center.x = prev_trackwindow.x;
	prev_center.y = prev_trackwindow.y;
	prev_image = image;
	return prev_trackwindow;
}

void CvFeatureTracker::setTrackingWindow(Rect _window)
{
	prev_trackwindow = _window;
}

Rect CvFeatureTracker::getTrackingWindow()
{
	return prev_trackwindow;
}


Point2f CvFeatureTracker::getTrackingCenter()
{
	return prev_center;
}
