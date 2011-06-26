#include <cv.h>
#include <cvaux.h>
#include <highgui.h>
#include <iostream>

#include "cvTracker.h"

using namespace cv;
using namespace std;

Mat image;
Rect selection;
Point origin;
bool selectObject = false;
int trackObject = 0;

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
        if( selection.width > 0 && selection.height > 0 )
            trackObject = -1;
        break;
    }
}


int main(int argc, char** argv) {
	CvTracker tracker;
	char img_file[20] = "../seqG/0001.png";

	namedWindow("Win", 1);
	setMouseCallback("Win", onMouse, 0);

	//tracker.priorImage(imread(img_file, CV_LOAD_IMAGE_COLOR));
	//tracker.addROI(Rect(135, 100, 50, 50));

	int hsize = 16;
    float hranges[] = {0,180};
    const float* phranges = hranges;
    Rect trackWindow;
	Mat hsv, mask, hue, hist, backproj;

	for(int i = 0; i < 1000; i++) {
		sprintf(img_file, "../seqG/%04d.png", i);
		image = imread(img_file, CV_LOAD_IMAGE_COLOR);

		if(!image.empty()) {
			cvtColor(image, hsv, CV_BGR2HSV);

			if(trackObject) {
				//image = tracker.updateTracks(image);
				inRange(hsv, Scalar(0, 30, MIN(10, 256)), Scalar(180, 256, MAX(10, 256)), mask);
				int ch[] = {0, 0};
				hue.create(hsv.size(), hsv.depth());
				mixChannels(&hsv, 1, &hue, 1, ch, 1);

				if( trackObject < 0 ) {
					Mat roi(hue, selection), maskroi(mask, selection);
					calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
					normalize(hist, hist, 0, 255, CV_MINMAX);

					trackWindow = selection;
					trackObject = 1;
				}

				calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
				backproj &= mask;
				RotatedRect trackBox = CamShift(backproj, trackWindow, TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));

                if( trackWindow.area() <= 1 )
                {
                    int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5)/6;
                    trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,
                                       trackWindow.x + r, trackWindow.y + r) &
                                  Rect(0, 0, cols, rows);
                }

				ellipse( image, trackBox, Scalar(0,0,255), 3, CV_AA );
			}

	        if( selectObject && selection.width > 0 && selection.height > 0 )
	        {
	            Mat roi(image, selection);
	            bitwise_not(roi, roi);
	        }

			imshow("Win", image);
			waitKey(30);
		}
	}

	return 0;
}

