// optical_flow.cpp : 定義主控台應用程式的進入點。
//

#include "stdafx.h"
#include <opencv2/opencv.hpp>

using namespace cv;

void drawOptFlowMap(const cv::Mat& flow,cv::Mat& cflowmap,int step,const cv::Scalar& color)
{
	for (int y = 0; y < cflowmap.rows; y += step)
		for (int x = 0; x < cflowmap.cols; x += step)
		{
			const Point2f& fxy = flow.at<cv::Point2f>(y, x);
			line(cflowmap,cv::Point(x, y),
			Point(cvRound(x + fxy.x), cvRound(y + fxy.y)),color);
			circle(cflowmap, cv::Point(x, y), 2, color, -1);
		}
}

int main() 
{

	VideoCapture cap("C:\\Users\\Jimmy\\Desktop\\MVI_7041.MOV"); // open the default camera
	if (!cap.isOpened())  // check if we succeeded
		return -1;

	Mat newFrame, newGray, prevGray;


	cap >> newFrame; // get a new frame from camera
	cvtColor(newFrame, newGray, CV_BGR2GRAY);
	prevGray = newGray.clone();

	double pyr_scale = 0.5;
	int levels = 3;
	int winsize = 5;
	int iterations = 5;
	int poly_n = 5;
	double poly_sigma = 1.1;
	int flags = 0;

	double fps = cap.get(CV_CAP_PROP_FPS);

	VideoWriter writer;
	writer = VideoWriter("test2.avi", CV_FOURCC('D', 'I', 'V', 'X'), fps, newGray.size());

	while (1) 
	{
		cap >> newFrame;
		if (newFrame.empty()) break;
		cvtColor(newFrame, newGray, CV_BGR2GRAY);

		Mat flow = Mat(newGray.size(), CV_32FC2);

		/* Do optical flow computation */
		calcOpticalFlowFarneback(prevGray,newGray,flow,pyr_scale,levels,winsize,iterations,poly_n,poly_sigma,flags);

		drawOptFlowMap(flow, newFrame, 20, CV_RGB(255, 0, 0));

		
		writer.write(newFrame);

		namedWindow("Output", WINDOW_AUTOSIZE);
		imshow("Output", newFrame);
		waitKey(1);

		prevGray = newGray.clone();
	}

	return 0;
}