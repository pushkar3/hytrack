#ifndef HYBRIDTRACKER_H_
#define HYBRIDTRACKER_H_

#include <cv.h>
#include <cvaux.h>
#include <iostream>

using namespace cv;
using namespace std;

class HybridTracker {
	enum tracker_t { TRACKER_MEANSHIFT, TRACKER_SIFT , TRACKER_SIFTRANSAC , TOTAL_TRACKERS};
	enum kernel_t { KERNEL_DISTANCE, KERNEL_NONE };
	float w[TOTAL_TRACKERS];
	Mat backProj[TOTAL_TRACKERS];

public:
	HybridTracker() { }
	~HybridTracker() { }

	void set(Size size) {

	}

	void useMeanShiftTracker(int use_distance_kernel) {

	}

	void useSIFTTracker(int use_distance_kernel, int use_ransac) {

	}

	void mergeTrackers() {
		normalize(backProj[0], backProj[0], 0, 1, NORM_L2);
	}

};


#endif /* HYBRIDTRACKER_H_ */
