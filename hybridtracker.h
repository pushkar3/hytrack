#ifndef HYBRIDTRACKER_H_
#define HYBRIDTRACKER_H_

#include <cv.h>
#include <cvaux.h>
#include <iostream>

using namespace cv;
using namespace std;

class HybridTracker {
public:
	enum tracker_t { TRACKER_MEANSHIFT, TRACKER_SIFT , TRACKER_SIFTRANSAC , TOTAL_TRACKERS};
	enum kernel_t { KERNEL_DISTANCE, KERNEL_NONE };
	float w[TOTAL_TRACKERS];
	Mat backproj[TOTAL_TRACKERS];

	Size _size;

public:
	HybridTracker() { }
	~HybridTracker() { }

	void set(Size size) {
		_size = size;
	}

	float getL2Norm(Point2d p1, Point2d p2) {
		float distance = (p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y);
		return distance;
	}

	Mat getDistanceProjection(Point2d center) {
		Mat hist(_size, CV_32F);
		for(int i = 0; i < hist.rows; i++) {
			for (int j = 0; j < hist.cols; j++) {
				hist.at<float>(i, j) = getL2Norm(Point(i, j), center);
			}
		}
		normalize(hist, hist, 255, 0, NORM_L2);
		return hist;
	}

	Mat getGaussianProjection() {

	}

	void mergeTrackers() {

	}

};


#endif /* HYBRIDTRACKER_H_ */
