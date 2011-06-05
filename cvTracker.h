/*
 * cvTracker.h
 *
 *  Created on: Jun 5, 2011
 *      Author: pushkar
 */

#ifndef CVTRACKER_H_
#define CVTRACKER_H_

class CvROITracker {
private:
	int w, h;
	CvSeq* x, y;		// Track of this object
	CvSeq* histogram;	// For Mean Shift
	CvSeq* features; 	// or use CvFeatureTree
						// SIFT features
	int w_ms;			// Param1 - Weight of shift based similarity measure
	int w_feature;		// Param2 - Weight of SOFT based similarity measure
public:
	CvROITracker(int x, int y, int w, int h);
	~CvROITracker();

	void update(CvMat* mat);
	// Uses some kernel to create a histogram
	// Creates a feature descriptor using some algorithms
	// Should the kernel and feature descriptor be specified by the user?
	// Or set optional params that will choose between various kernels and feature descriptors.

	void predict();
	void estimate();
	// Parameter estimation using EM

	void smooth();
	// Post processing of the path using Kalman filter

};

class CvTracker {
private:
	CvSeq* tracks;		// Vector of CvROI that are being tracked
public:
	CvTracker() { };
	~CvTracker() { };

	void addObject(CvROITracker _roi) {
		// Add _roi to tracks
	}

	void deleteObject(CvROITracker _roi) {
		// remove _roi from tracks
	}

	CvROITracker* getTrackedObjects();

	void updateTracks(CvMat* mat) {
		/* foreach object in track
		_roi.update(mat);
		_roi.predict();
		_roi.estimate();
		_roi.smooth();
		*/
		// end foreach
	}
};

#endif /* CVTRACKER_H_ */
