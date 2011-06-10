#include <cv.h>
#include <cvaux.h>
#include <highgui.h>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
	Mat img1;
	FeatureDetector* detector = 0;
	DescriptorExtractor* descriptor = 0;
	vector<KeyPoint> keypoints;
	DescriptorMatcher* matcher = 0;
	vector<DMatch> matches;
	vector<Mat> desc_vector;

	img1 = imread("ex.jpg", CV_LOAD_IMAGE_COLOR);
	if (img1.empty()) {
		cout << "Can not read image" << endl;
		return 0;
	}

	detector = new SiftFeatureDetector(SIFT::DetectorParams::GET_DEFAULT_THRESHOLD(),
			SIFT::DetectorParams::GET_DEFAULT_EDGE_THRESHOLD(),
			SIFT::CommonParams::AVERAGE_ANGLE);
	// detector = new SurfFeatureDetector( 400./*hessian_threshold*/, 3 /*octaves*/, 4/*octave_layers*/);

	descriptor = new SurfDescriptorExtractor(3/*octaves*/, 4/*octave_layers*/, false/*extended*/);

	detector->detect(img1, keypoints);
	cout << "Number of keypoints: " << keypoints.size() << endl;

	Mat desc;
	if(keypoints.size() > 0) {
		descriptor->compute(img1, keypoints, desc);
		desc_vector.push_back(desc);
	}

	for(int i = 0; i < keypoints.size(); i++) {
		circle(img1, keypoints[i].pt, 2, CV_RGB(0, 255, 0));
	}

	matcher = new BruteForceMatcher<L2<float> > ();

	matcher->clear();
	matcher->add(desc_vector);
	matcher->match(desc, matches);

	namedWindow("Win", 1);
	imshow("Win", img1);
	waitKey(0);

	return 0;
}
