#include <cv.h>
#include <ml.h>
#include <cvaux.h>
#include <highgui.h>
#include <iostream>

#include "colortracker.h"
#include "featuretracker.h"

using namespace cv;
using namespace std;

Mat image;
Rect selection;
Point origin;
bool selectObject = false;
int trackObject = 0;

ColorTracker ctracker;
FeatureTracker ftracker;

void onMouse(int event, int x, int y, int, void*) {
    if(selectObject) {
        selection.x = MIN(x, origin.x);
        selection.y = MIN(y, origin.y);
        selection.width = std::abs(x - origin.x);
        selection.height = std::abs(y - origin.y);
        selection &= Rect(0, 0, image.cols, image.rows);
    }

    switch(event) {
    case CV_EVENT_LBUTTONDOWN:
        origin = Point(x,y);
        selection = Rect(x,y,0,0);
        selectObject = true;
        break;
    case CV_EVENT_LBUTTONUP:
        selectObject = false;
        ctracker.init(image, selection);
        ftracker.init(image, selection);
        trackObject = 1;
        break;
    }
}

float w1 = 0.5;
float w2 = 0.5;

int main(int argc, char** argv) {

	Mat mask;

	char img_file[20] = "seqG/0001.png";
	namedWindow("Temp", 1);
	namedWindow("Win", 1);
	setMouseCallback("Win", onMouse, 0);

	image = imread(img_file, CV_LOAD_IMAGE_COLOR);
	mask = Mat::zeros(image.size(), CV_8UC1);

	for(int i = 0; i < 1000; i++) {
		sprintf(img_file, "seqG/%04d.png", i);
		image = imread(img_file, CV_LOAD_IMAGE_COLOR);

		if (!image.empty()) {

			if(trackObject) {
				Rect win1 = ctracker.track(image);
				Rect win2 = ftracker.track(image);
				Rect win3 = win1;

//				double variance = (win1.x-win2.x)*(win1.x-win2.x) + (win1.y-win2.y)*(win1.y-win2.y);
//				variance = sqrt(variance)/100.0f;
//				cout << "variance is " << variance << endl;

				mask = Scalar(0);
				rectangle(mask, Point(win1.x, win1.y), Point(win1.x + win1.width, win1.y + win1.height), 255, CV_FILLED);
				rectangle(mask, Point(win2.x, win2.y), Point(win2.x + win2.width, win2.y + win2.height), 255, CV_FILLED);
				//circle(mask, Point(win2.x, win2.y), variance, 255, CV_FILLED);

				//rectangle(mask, Point(100, 100), Point(200, 200), 255, CV_FILLED);
				//circle(mask, Point(100, 100), 10, 255, CV_FILLED);

				cout << endl;
				int sample_cnt = 0;
				for(int j = 0; j < mask.rows; j++) {
					for(int k = 0; k < mask.cols; k++) {
						if(mask.at<unsigned char>(j, k) != 0) {
							sample_cnt++;
						}
					}
				}
				cout << sample_cnt << endl;

				CvMat* samples = cvCreateMat(sample_cnt, 2, CV_32FC1);
				CvMat* labels = cvCreateMat(sample_cnt, 1, CV_32SC1);
				CvEM em_model;
				CvEMParams params;

				params.covs = NULL;
				params.means = NULL;
				params.weights = NULL;
				params.probs = NULL;
				params.nclusters = 1;
				params.cov_mat_type = CvEM::COV_MAT_SPHERICAL;
				params.start_step = CvEM::START_AUTO_STEP;
				params.term_crit.max_iter = 10;
				params.term_crit.epsilon = 0.1;
				params.term_crit.type = CV_TERMCRIT_ITER|CV_TERMCRIT_EPS;

				int c = 0;
				for(int j = 0; j < image.rows; j++) {
					for(int k = 0; k < image.cols; k++) {
						if(mask.at<int>(j, k) != 0 && c < sample_cnt) {
							samples->data.fl[c*2] = j;
							samples->data.fl[c*2+1] = k;
							c++;
						}
					}
				}

				em_model.train(samples, 0, params, labels);
			    for(int i = 0; i < em_model.get_nclusters(); i++) {
			    	cout << "Mean " << i << " is " << em_model.getMeans().at<double>(i, 0) << ", " << em_model.getMeans().at<double>(i, 1) << endl;
			    	circle(mask, Point(em_model.getMeans().at<double>(i, 0), em_model.getMeans().at<double>(i, 1)), 4, 127, -1);
			    }

			    cout << sample_cnt << endl;

				imshow("Temp", mask);
				imshow("Win", image);
				//waitKey(0);

				win3.x = em_model.getMeans().at<double>(0, 0);
				win3.y = em_model.getMeans().at<double>(0, 1);
				rectangle(image, Point(win3.x, win3.y), Point(win3.x + win3.width, win3.y + win3.height), Scalar(0, 255, 0), 2, CV_AA);

				ctracker.setTrackWindow(win3);
				ftracker.setTrackWindow(win3);

				cvReleaseMat(&samples);
				cvReleaseMat(&labels);
			}

			if (selectObject && selection.width > 0 && selection.height > 0) {
				Mat roi(image, selection);
				bitwise_not(roi, roi);
			}


			sprintf(img_file, "out/%04d.png", i);
			imwrite(img_file, image);
			imshow("Win", image);
			waitKey(30);
		}
		else
			i =0;
	}

	return 0;
}

