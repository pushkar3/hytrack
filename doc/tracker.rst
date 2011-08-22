Hybrid Tracking
===============

.. highlight:: cpp

A hybrid tracker combines estimates from two tracking algorithms for robust tracking.

The hybrid tracker uses MeanShift at its base. The MeanShift is a gradient descent technique that tries to find the best back projection. Like other optimization techniques, MeanShift can get stuck in local minimas or might require a lot of iterations to end up in the global minimum. To overcome this, hybrid trackers use various techniques. [Zhou09]_ use SIFT correspondence with MeanShift for tracking. [Avidan05]_ call this method ensemble tracking, where weak classifiers, which are trained online, are used alongwith MeanShift.

MotionModel
------------
Defines the motion model for tracking. Defines parameters for that model.

.. ocv:function:: MotionModel::MotionModel(int model = LOW_PASS_FILTER, float low_pass_gain = 0.1)
Defines the motion model as a simple Low Pass Filter. 

.. ocv:function:: MotionModel::MotionModel(int model = EM, CvEMParams em_params = CvEMParams())
Defines the motion model with EM. Works well with occlusions.


MeanShiftTrackerParams
---------------------
Defines parameters for the Mean Shift Tracker

.. ocv:function:: MeanShiftTrackerParams::MeanShiftTrackerParams(int tracking_type = CvMeanShiftTrackerParams::HS, CvTermCriteria term_crit = CvTermCriteria()
	:param tracking_type: Color channels to be used for tracking. Possible values are:
		* **CvMeanShiftTrackerParams::H** Only use the H channel.
		* **CvMeanShiftTrackerParams::HS** Only use the HS channels.
		* **CvMeanShiftTrackerParams::HSV** Only use the HSV channels, rarely used.	
    
    :param term_crit: Termination criteria for the Cam shift algorithm.
    
    :param h_range[]: An array specifying what range of H will be used.
    :param s_range[]: An array specifying what range of S will be used.
    :param v_range[]: An array specifying what range of V will be used.

FeatureTrackerParams
---------------------
Defines parameters for the Feature Tracker

.. ocv:function:: FeatureTrackerParams::FeatureTrackerParams(int feature_type = 0, int window_size = 0)

	:param feature_type: Features to be used for tracking. Possible values are:
		* **CvFeatureTrackerParams::SIFT** Use SIFT with default parameters.
		* **CvFeatureTrackerParams::SURF** Use SURF with default parameters.	
    
    :param window_size: Window size in pixels to be considered for searching.
    
HybridTrackerParams
---------------------
Defines parameters for the Hybrid Tracker

.. ocv:function:: HybridTrackerParams::HybridTrackerParams(float ft_tracker_weight = 0.5, float ms_tracker_weight = 0.5,
			CvFeatureTrackerParams ft_params = CvFeatureTrackerParams(),
			CvMeanShiftTrackerParams ms_params = CvMeanShiftTrackerParams(),
			CvMotionModel model = CvMotionModel(CvMotionModel::LOW_PASS_FILTER, 0.1)))

	:param ft_tracker_weight: Initial importance of the feature tracker (< 1)
	:param ms_tracker_weight: Initial importance of the mean shift tracker (< 1)
	:param ft_params: FeatureTrackerParams for the feature tracking algorithm
	:param ms_params: MeanShiftTrackerParams for the meanshift tracking algorithm
	:param model: MotionModel for the tracking algorithm
	
MeanShiftTracker
-----------------
.. ocv:class:: MeanShiftTracker

Tracking using CamShift

.. ocv:function:: MeanShiftTracker::MeanShiftTracker(CvMeanShiftTrackerParams _params = CvMeanShiftTrackerParams())

MeanShiftTracker::newTrackingWindow
------------------------------------
Defines a new tracking window

.. ocv:function:: MeanShiftTracker::newTrackingWindow(Mat image, Rect selection)

MeanShiftTracker::updateTrackingWindow
--------------------------------------
Updates the tracker

.. ocv:function:: RotatedRect MeanShiftTracker::updateTrackingWindow(Mat image)

MeanShiftTracker::getHistogramProjection
---------------------------------------
Gives a probability distribution graph of where it thinks the object is

.. ocv:function:: Mat MeanShiftTracker::getHistogramProjection(int type)

MeanShiftTracker::setTrackingWindow
------------------------------------
Sets the tracking window

.. ocv:function:: void MeanShiftTracker::setTrackingWindow(Rect _window)

MeanShiftTracker::getTrackingWindow
------------------------------------
Gets the tracking window

.. ocv:function:: Rect MeanShiftTracker::getTrackingWindow()

MeanShiftTracker::getTrackingCenter
------------------------------------
Gets the tracking center

.. ocv:function:: Point2f MeanShiftTracker::getTrackingCenter()

FeatureTracker
-----------------
.. ocv:class:: FeatureTracker

Tracking using SIFT/SURF features

.. ocv:function:: FeatureTracker::FeatureTracker(CvFeatureTrackerParams _params = CvFeatureTrackerParams())

FeatureTracker::newTrackingWindow
------------------------------------
Defines a new tracking window

.. ocv:function:: FeatureTracker::newTrackingWindow(Mat image, Rect selection)

FeatureTracker::updateTrackingWindow
--------------------------------------
Updates the tracker

.. ocv:function:: Rect FeatureTracker::updateTrackingWindow(Mat image)

FeatureTracker::setTrackingWindow
------------------------------------
Sets the tracking window

.. ocv:function:: void FeatureTracker::setTrackingWindow(Rect _window)

FeatureTracker::getTrackingWindow
------------------------------------
Gets the tracking window

.. ocv:function:: Rect FeatureTracker::getTrackingWindow()

FeatureTracker::getTrackingCenter
------------------------------------
Gets the tracking center

.. ocv:function:: Point2f FeatureTracker::getTrackingCenter()

	
HybridTracker
--------------
.. ocv:class:: HybridTracker
Computes a probability density function from the region of interest selected and tracks it. The tracker consists of two or more trackers whose results are used in an EM framework.

.. ocv:function:: HybridTracker::HybridTracker(CvHybridTrackerParams _params = CvHybridTrackerParams())

MeanShiftTracker::newTracker
------------------------------------
Defines a new tracking window

.. ocv:function:: HybridTracker::newTrackingWindow(Mat image, Rect selection)

HybridTracker::updateTracker
--------------------------------------
Updates the tracker using the parameters from the motion model. It can either use a low pass filter or EM.

.. ocv:function:: void HybridTracker::updateTrackingWindow(Mat image)

The function implements a hybrid tracker as given in [Zhou09]_.



.. [Zhou09] Zhou, H. and Yuan, Y. and Shi, C.. Object tracking using SIFT features and mean shift.

... [Avidan05] Avidan, S.. Ensemble tracking.
