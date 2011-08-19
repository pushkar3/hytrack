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

CvMeanShiftTracker::CvMeanShiftTracker()
{
}

CvMeanShiftTracker::~CvMeanShiftTracker()
{
}

void CvMeanShiftTracker::init(Mat image, Rect selection)
{

	hist.release();
	int histSize = 16;
	int channels[] =
	{ 0, 0 };
	float hrange[] =
	{ 0, 180 };
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

RotatedRect CvMeanShiftTracker::track(Mat image)
{
	int channels[] =
	{ 0, 0 };
	float hrange[] =
	{ 0, 180 };
	const float* ranges = hrange;

	cvtColor(image, hsv, CV_BGR2HSV);
	inRange(hsv, Scalar(0, 30, MIN(10, 256)), Scalar(180, 256, MAX(10, 256)),
			mask);
	mixChannels(&hsv, 1, &hue, 1, channels, 1);
	calcBackProject(&hue, 1, 0, hist, backproj, &ranges);
	//normalize(backproj, backproj, 0, 255, CV_MINMAX);
	backproj &= mask;
	// meanShift(backproj, trackwindow, TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1));
	RotatedRect trackbox = CamShift(backproj, trackwindow, TermCriteria(
			CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1));
	int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5)
			/ 6;
	trackwindow = Rect(trackwindow.x - r, trackwindow.y - r, trackwindow.x + r,
			trackwindow.y + r) & Rect(0, 0, cols, rows);
	center.x = trackwindow.x + trackwindow.width / 2;
	center.y = trackwindow.y + trackwindow.height / 2;
	return trackbox;

}
