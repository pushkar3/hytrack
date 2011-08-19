Hybrid Tracking
===============

.. highlight:: cpp



MeanShiftTrackerParams
---------------------
Defines parameters for the Mean Shift Tracker

FeatureTrackerParams
---------------------
Defines parameters for the Feature Tracker

HybridTrackerParams
---------------------
Defines parameters for the Hybrid Tracker


HybridTracker
--------------
Computes a probability density function from the region of interest selected and tracks it. The tracker consists of two or more trackers whose results are used in an EM framework.

The function implements a hybrid tracker as given in [Zhou09]_.

.. [Zhou09] Zhou, H. and Yuan, Y. and Shi, C.. Object tracking using SIFT features and mean shift.
