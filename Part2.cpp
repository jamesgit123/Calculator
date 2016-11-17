/*
Copyright (c) 2014, Nghia Ho
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>
#include <time.h>

using namespace std;
using namespace cv;

// This video stablisation smooths the global trajectory using a sliding average window

const int SMOOTHING_RADIUS = 30; // In frames. The larger the more stable the video, but less reactive to sudden panning
const int HORIZONTAL_BORDER_CROP = 20; // In pixels. Crops the border to reduce the black borders from stabilisation being too noticeable.
const int SEC_IN_MILLISEC = 600;

// 1. Get previous to current frame transformation (dx, dy, da) for all frames
// 2. Accumulate the transformations to get the image trajectory
// 3. Smooth out the trajectory using an averaging window
// 4. Generate new set of previous to current transform, such that the trajectory ends up being the same as the smoothed trajectory
// 5. Apply the new transformation to the video

struct TransformParam
{
    TransformParam() {}
    TransformParam(double _dx, double _dy, double _da) {
        dx = _dx;
        dy = _dy;
        da = _da;
    }

    double dx;
    double dy;
    double da; // angle
};

struct Trajectory
{
    Trajectory() {}
    Trajectory(double _x, double _y, double _a) {
        x = _x;
        y = _y;
        a = _a;
    }

    double x;
    double y;
    double a; // angle
};

int tick(){
	return clock();
}

float tock(int initTime){
	return (float)(clock() - initTime)/CLOCKS_PER_SEC;
}

// ######################################################################################################################### //
// ######################################################################################################################### //
// ######################################################################################################################### //

void showImage(Mat cur, Mat cur2){
	// Now draw the original and stablised side by side for coolness
	Mat canvas = Mat::zeros(cur.rows, cur.cols*2+10, cur.type());

	cur.copyTo(canvas(Range::all(), Range(0, cur2.cols)));
	cur2.copyTo(canvas(Range::all(), Range(cur2.cols+10, cur2.cols*2+10)));

	// If too big to fit on the screen, then scale it down by 2, hopefully it'll fit :)
	if(canvas.cols > 1920) {
		resize(canvas, canvas, Size(canvas.cols/2, canvas.rows/2));
	}

	imshow("before and after", canvas);
}


Mat applyTransformation(Mat cur, TransformParam transform, int vert_border){
	Mat T(2,3,CV_64F);

	T.at<double>(0,0) = cos(transform.da);
	T.at<double>(0,1) = -sin(transform.da);
	T.at<double>(1,0) = sin(transform.da);
	T.at<double>(1,1) = cos(transform.da);

	T.at<double>(0,2) = transform.dx;
	T.at<double>(1,2) = transform.dy;

	Mat cur2;

	warpAffine(cur, cur2, T, cur.size());

	cur2 = cur2(Range(vert_border, cur2.rows-vert_border), Range(HORIZONTAL_BORDER_CROP, cur2.cols-HORIZONTAL_BORDER_CROP));

	// Resize cur2 back to cur size, for better side by side comparison
	resize(cur2, cur2, cur.size());

	return cur2;
}

Trajectory getTrajectory(Trajectory prevTrajectory, TransformParam tp){
	double x,y,a;
	x = prevTrajectory.x + tp.dx;
	y =  prevTrajectory.y + tp.dy;
	a =  prevTrajectory.a + tp.da;

	return Trajectory(x,y,a);
}

Trajectory getSmoothedTrajectory(vector<Trajectory> trajectory, int i){
	double sumX, sumY, sumA;
	int count=0;
	int trajSize = trajectory.size();
	sumX = sumY = sumA = 0;

	for(int j=-SMOOTHING_RADIUS; j <= SMOOTHING_RADIUS; j++){
		if(i+j >= 0 && i+j < trajSize) {
			sumX += trajectory[i+j].x;
			sumY += trajectory[i+j].y;
			sumA += trajectory[i+j].a;

			count++;
		}
	}

	double avg_x = sumX / count;
	double avg_y = sumY / count;
	double avg_a = sumA / count;

	return Trajectory(avg_x, avg_y, avg_a);
}

TransformParam getSmoothedTransform(TransformParam prevTranform, Trajectory smoothedTrajectory, double x, double y, double a){

	// target - current
	double diff_x = smoothedTrajectory.x - x;
	double diff_y = smoothedTrajectory.y - y;
	double diff_a = smoothedTrajectory.a - a;

	double dx = prevTranform.dx + diff_x;
	double dy = prevTranform.dy + diff_y;
	double da = prevTranform.da + diff_a;
	return TransformParam(dx,dy,da);
}
// ######################################################################################################################### //
// ######################################################################################################################### //
// ######################################################################################################################### //

int main(int argc, char **argv)
{
    if(argc < 2) {
        cout << "./VideoStab [video.avi]" << endl;
        return 0;
    }

    // For further analysis
    ofstream out_transform("prev_to_cur_transformation.txt");
    ofstream out_trajectory("trajectory.txt");
    ofstream out_smoothed_trajectory("smoothed_trajectory.txt");
    ofstream out_new_transform("new_prev_to_cur_transformation.txt");

    cout << argv[1] << endl;

    VideoCapture cap(argv[1]);
    assert(cap.isOpened());
    int initTime = tick();

    int fps = cap.get(CV_CAP_PROP_FPS);
    cout << fps << endl;

    Mat cur, cur_grey;
    Mat prev, prev_grey;

    cap >> prev;
    cvtColor(prev, prev_grey, COLOR_BGR2GRAY);

    // Step 1 - Get previous to current frame transformation (dx, dy, da) for all frames
    vector <TransformParam> prev_to_cur_transform; // previous to current

    int k=1;
    int max_frames = cap.get(CV_CAP_PROP_FRAME_COUNT);
    Mat last_T;

    while(true) {
        cap >> cur;

        if(cur.data == NULL) {
            break;
        }

        cvtColor(cur, cur_grey, COLOR_BGR2GRAY);

        // vector from prev to cur
        vector <Point2f> prev_corner, cur_corner;
        vector <Point2f> prev_corner2, cur_corner2;
        vector <uchar> status;
        vector <float> err;

        goodFeaturesToTrack(prev_grey, prev_corner, 200, 0.01, 30);
        calcOpticalFlowPyrLK(prev_grey, cur_grey, prev_corner, cur_corner, status, err);

        // weed out bad matches
        for(size_t i=0; i < status.size(); i++) {
            if(status[i]) {
                prev_corner2.push_back(prev_corner[i]);
                cur_corner2.push_back(cur_corner[i]);
            }
        }

        // translation + rotation only
        Mat T = estimateRigidTransform(prev_corner2, cur_corner2, false); // false = rigid transform, no scaling/shearing

        // in rare cases no transform is found. We'll just use the last known good transform.
        if(T.data == NULL) {
            last_T.copyTo(T);
        }

        T.copyTo(last_T);

        // decompose T
        double dx = T.at<double>(0,2);
        double dy = T.at<double>(1,2);
        double da = atan2(T.at<double>(1,0), T.at<double>(0,0));

        prev_to_cur_transform.push_back(TransformParam(dx, dy, da));

        out_transform << k << " " << dx << " " << dy << " " << da << endl;

        cur.copyTo(prev);
        cur_grey.copyTo(prev_grey);

        cout << "Time: " << tock(initTime) << " seconds " << "Frame: " << k << "/" << max_frames << " - good optical flow: " << prev_corner2.size() << endl;
        k++;
    }

    // Step 2 - Accumulate the transformations to get the image trajectory

    // Accumulated frame to frame transform
    double a = 0;
    double x = 0;
    double y = 0;

    vector <Trajectory> trajectory; // trajectory at all frames
    Trajectory tempTrajectory;

    for(size_t i=0; i < prev_to_cur_transform.size(); i++) {
    	TransformParam tp = prev_to_cur_transform[i];
    	tempTrajectory = getTrajectory(tempTrajectory,tp);
    	trajectory.push_back(tempTrajectory);
        out_trajectory << (i+1) << " " << x << " " << y << " " << a << endl;
    }

    // Step 3 - Smooth out the trajectory using an averaging window
    vector <Trajectory> smoothed_trajectory; // trajectory at all frames
    Trajectory smoothedTempTrajectory;
    int trajSize = trajectory.size();

    for(int i=0; i < trajSize; i++) {
        smoothedTempTrajectory = getSmoothedTrajectory(trajectory, i);
        smoothed_trajectory.push_back(smoothedTempTrajectory);
    }

    // Step 4 - Generate new set of previous to current transform, such that the trajectory ends up being the same as the smoothed trajectory
    vector <TransformParam> new_prev_to_cur_transform;

    // Accumulated frame to frame transform
    a = 0;
    x = 0;
    y = 0;
    TransformParam tempNewTransform;

    for(size_t i=0; i < prev_to_cur_transform.size(); i++) {
        x += prev_to_cur_transform[i].dx;
        y += prev_to_cur_transform[i].dy;
        a += prev_to_cur_transform[i].da;

        tempNewTransform = getSmoothedTransform(prev_to_cur_transform[i], smoothed_trajectory[i], x,y,a);

        new_prev_to_cur_transform.push_back(tempNewTransform);

        //out_new_transform << (i+1) << " " << dx << " " << dy << " " << da << endl;
    }

    // Step 5 - Apply the new transformation to the video
    cap.set(CV_CAP_PROP_POS_FRAMES, 0);


    int vert_border = HORIZONTAL_BORDER_CROP * prev.rows / prev.cols; // get the aspect ratio correct
    int delay = (int)(SEC_IN_MILLISEC/fps);

    k=0;
    initTime = tick();
    Mat cur2;
    while(k < max_frames-1) { // don't process the very last frame, no valid transform
        cap >> cur;

        if(cur.data == NULL) {
            break;
        }

        cur2 = applyTransformation(cur, new_prev_to_cur_transform[k], vert_border);

        showImage(cur, cur2);

        waitKey(delay);

        k++;
    }
    cout << "Video Duration: " << tock(initTime) << endl;

    return 0;
}
