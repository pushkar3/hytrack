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
#include "hybridtracker.hpp"

using namespace cv;

CvFeatureTracker::CvFeatureTracker()
{
	detector = new SiftFeatureDetector(
			SIFT::DetectorParams::GET_DEFAULT_THRESHOLD(),
			SIFT::DetectorParams::GET_DEFAULT_EDGE_THRESHOLD(),
			SIFT::CommonParams::AVERAGE_ANGLE);
	// detector = new SurfFeatureDetector(400, 3, 4);


	descriptor = new SurfDescriptorExtractor(3, 4, false);

	matcher = new BruteForceMatcher<L2<float> > ();
	// SSD matcher
}


CvFeatureTracker::~CvFeatureTracker()
{
}

void CvFeatureTracker::init(Mat image, Rect selection)
{
	trackWindow = selection;
	prev_image = image;

	Mat mask = Mat::zeros(image.size(), CV_8UC1);
	rectangle(mask, Point(trackWindow.x, trackWindow.y), Point(trackWindow.x
			+ trackWindow.width, trackWindow.y + trackWindow.height), Scalar(
			255), CV_FILLED);

	prev_desc_vector.clear();
	detector->detect(prev_image, prev_keypoints, mask);

	if (prev_keypoints.size() > 0)
	{
		descriptor->compute(prev_image, prev_keypoints, prev_desc);
	}
}

void CvFeatureTracker::setTrackWindow(Rect _window)
{
	trackWindow = _window;
}

Rect CvFeatureTracker::track(Mat image)
{
	init(prev_image, trackWindow);

	vector<KeyPoint> current_keypoints;
	Mat current_desc;

	int windowSize = 10;
	Rect window(trackWindow.x - windowSize, trackWindow.y - windowSize,
			trackWindow.width + windowSize, trackWindow.height + windowSize);

	Mat mask = Mat::zeros(image.size(), CV_8UC1);
	rectangle(mask, Point(window.x, window.y), Point(window.x + window.width,
			window.y + window.height), Scalar(255), CV_FILLED);

	detector->detect(image, current_keypoints, mask);

	if (current_keypoints.size() > 4)
	{
		descriptor->compute(image, current_keypoints, current_desc);
		matcher->match(prev_desc, current_desc, matches);

		Point p0 = prev_keypoints[matches[0].trainIdx].pt;
		Point n0 = current_keypoints[matches[0].queryIdx].pt;

#if 0
		Point p1 = prev_keypoints[matches[1].trainIdx].pt;
		Point n1 = current_keypoints[matches[1].queryIdx].pt;

		double dp = sqrt((p0.x - p1.x)*(p0.x - p1.x) + (p0.y - p1.y)*(p0.y - p1.y));
		double dn = sqrt((n0.x - n1.x)*(n0.x - n1.x) + (n0.y - n1.y)*(n0.y - n1.y));
		printf("scale: %lf, %lf\n", dn, dp);
		double scale = dn/dp;
		trackWindow.width *= scale;
		trackWindow.height *= scale;
#endif


		trackWindow.x += (p0.x - n0.x);
		trackWindow.y += (p0.y - n0.y);
	}

	return trackWindow;
}

