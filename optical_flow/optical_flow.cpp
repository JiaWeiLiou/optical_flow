// optical_flow.cpp : 定義主控台應用程式的進入點。
//

#include "stdafx.h"
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <time.h>

using namespace std;
using namespace cv;

#define UNKNOWN_FLOW_THRESH 1e9 

void drawOptFlowMap(const Mat& oldFlow, Mat& flow, Mat& cflowmap, int step, const Scalar& color)
{
	for (int y = step; y < cflowmap.rows; y += step)
		for (int x = step; x < cflowmap.cols; x += step)
		{
			Point2f& fxy = flow.at<Point2f>(y, x);
			const Point2f& oldFxy = oldFlow.at<Point2f>(y, x);
			float avgFx = 0;
			float avgFy = 0;
			float avgOldFx = 0;
			float avgOldFy = 0;
			int avgWin = 0;

			if ((step / 3) % 2 == 1)
				avgWin = step / 3;
			else
				avgWin = (step / 3) + 1;

			for (int j = -avgWin / 2; j < avgWin / 2; j++)
				for (int i = -avgWin / 2; i < avgWin / 2; i++)
				{
					avgFx += flow.at<Point2f>(y + j, x + i).x / (avgWin *avgWin);
					avgFy += flow.at<Point2f>(y + j, x + i).y / (avgWin *avgWin);
					avgOldFx += oldFlow.at<Point2f>(y + j, x + i).x / (avgWin *avgWin);
					avgOldFy += oldFlow.at<Point2f>(y + j, x + i).y / (avgWin *avgWin);
				}

			//鄰近區域平均方向及平均長度拘束
			if ((fxy.x*avgFx < 0 || fxy.y*avgFy  < 0) || (abs(fxy.x / avgFx)>2) || (abs(fxy.y / avgFy)>2))
			{
				fxy.x = avgFx;
				fxy.y = avgFy;
			}

			//上一幀平均方向拘束
			if ((fxy.x*avgOldFx < 0 || fxy.y*avgOldFy < 0) || (abs(fxy.x / avgOldFx)>2) || (abs(fxy.y / avgOldFy)>2))
			{
				fxy.x = avgOldFx;
				fxy.y = avgOldFy;
			}

			line(cflowmap, Point(x, y), Point(cvRound(x + fxy.x * 5), cvRound(y + fxy.y * 5)), color);
			circle(cflowmap, Point(x, y), 2, color, -1);
		}
}

void makecolorwheel(vector<Scalar> &colorwheel)
{
	int RY = 15;
	int YG = 6;
	int GC = 4;
	int CB = 11;
	int BM = 13;
	int MR = 6;

	int i;

	for (i = 0; i < RY; i++) colorwheel.push_back(Scalar(255, 255 * i / RY, 0));
	for (i = 0; i < YG; i++) colorwheel.push_back(Scalar(255 - 255 * i / YG, 255, 0));
	for (i = 0; i < GC; i++) colorwheel.push_back(Scalar(0, 255, 255 * i / GC));
	for (i = 0; i < CB; i++) colorwheel.push_back(Scalar(0, 255 - 255 * i / CB, 255));
	for (i = 0; i < BM; i++) colorwheel.push_back(Scalar(255 * i / BM, 0, 255));
	for (i = 0; i < MR; i++) colorwheel.push_back(Scalar(255, 0, 255 - 255 * i / MR));
}

void motionToColor(Mat flow, Mat &color)
{
	if (color.empty())
		color.create(flow.rows, flow.cols, CV_8UC3);

	static vector<Scalar> colorwheel; //Scalar r,g,b  
	if (colorwheel.empty())
		makecolorwheel(colorwheel);

	// determine motion range:  
	float maxrad = -1;

	// Find max flow to normalize fx and fy  
	for (int i = 0; i < flow.rows; ++i)
	{
		for (int j = 0; j < flow.cols; ++j)
		{
			Vec2f flow_at_point = flow.at<Vec2f>(i, j);
			float fx = flow_at_point[0];
			float fy = flow_at_point[1];
			if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))
				continue;
			float rad = sqrt(fx * fx + fy * fy);
			maxrad = maxrad > rad ? maxrad : rad;
		}
	}

	for (int i = 0; i < flow.rows; ++i)
	{
		for (int j = 0; j < flow.cols; ++j)
		{
			uchar *data = color.data + color.step[0] * i + color.step[1] * j;
			Vec2f flow_at_point = flow.at<Vec2f>(i, j);

			float fx = flow_at_point[0] / maxrad;
			float fy = flow_at_point[1] / maxrad;
			if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))
			{
				data[0] = data[1] = data[2] = 0;
				continue;
			}
			float rad = sqrt(fx * fx + fy * fy);

			float angle = atan2(-fy, -fx) / CV_PI;
			float fk = (angle + 1.0) / 2.0 * (colorwheel.size() - 1);
			int k0 = (int)fk;
			int k1 = (k0 + 1) % colorwheel.size();
			float f = fk - k0;
			//f = 0; // uncomment to see original color wheel  

			for (int b = 0; b < 3; b++)
			{
				float col0 = colorwheel[k0][b] / 255.0;
				float col1 = colorwheel[k1][b] / 255.0;
				float col = (1 - f) * col0 + f * col1;
				if (rad <= 1)
					col = 1 - rad * (1 - col); // increase saturation with radius  
				else
					col *= .75; // out of range  
				data[2 - b] = (int)(255.0 * col);
			}
		}
	}
}

void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step, const Scalar& color)
{
	for (int y = step; y < cflowmap.rows; y += step)
		for (int x = step; x < cflowmap.cols; x += step)
		{
			const Point2f& fxy = flow.at<Point2f>(y, x);
			line(cflowmap, Point(x, y), Point(cvRound(x + fxy.x * 5), cvRound(y + fxy.y * 5)), color);
			circle(cflowmap, Point(x, y), 2, color, -1);
		}
}

int main()
{
	string infile;
	string outfile;
	cout << "Please enter infile : ";
	//cin >> infile;
	infile = "C:\\Users\\Jimmy\\Desktop\\研究\\渠道影片\\DSC_0007_cut.avi";

	/*確認檔案是否存在*/
	VideoCapture cap(infile); // open the default camera
	if (!cap.isOpened())  // check if we succeeded
	{
		cout << "Error opening Video !" << endl;
		system("pause");
		return -1;
	}

	/*計算程式執行時間*/
	double timeStart, timeEnd;
	timeStart = clock();

	/*轉換前後座標*/
	Point2f beforept[4] = { Point2f(767,267),Point2f(1463,307),Point2f(1595,977),cv::Point2f(569,900) };
	Point2f afterpt[4] = { Point2f(0,0),Point2f(660,0),Point2f(660,660),cv::Point2f(0,660) };
	Size aftersize = Size(afterpt[2].x, afterpt[2].y);

	//修改文件名
	int pos1 = infile.find_last_of('/\\');
	int pos2 = infile.find_last_of('.');
	string filepath(infile.substr(0, pos1));
	string infile_name(infile.substr(pos1 + 1, pos2 - pos1 - 1));
	outfile = filepath + "\\" + infile_name + "_cut.avi";

	Mat newFrame, newGray, newGray_temp, newGray_temp_temp, prevGray;

	cap >> newFrame; // get a new frame from camera
	cvtColor(newFrame, newGray_temp, CV_BGR2GRAY);

	/*透視投影轉換*/
	Mat perspective_matrix = getPerspectiveTransform(beforept, afterpt);
	warpPerspective(newGray_temp, newGray, perspective_matrix, aftersize);
	threshold(newGray, newGray_temp_temp, 175, 255, THRESH_TOZERO);

	prevGray = newGray_temp_temp.clone();

	double pyr_scale = 0.5;
	int levels = 3;
	int winsize = 5;
	int iterations = 5;
	int poly_n = 5;
	double poly_sigma = 1.1;
	int flags = OPTFLOW_USE_INITIAL_FLOW;

	double fps = cap.get(CV_CAP_PROP_FPS);

	VideoWriter writer = VideoWriter(outfile, CV_FOURCC('D', 'I', 'V', 'X'), fps, newGray.size(), 0);

	int test = 0;
	Mat flow = Mat(newGray.size(), CV_32FC2);
	Mat oldFlow = Mat(newGray.size(), CV_32FC2);

	while (1)
	{
		++test;

		cap >> newFrame;

		if (newFrame.empty()) break;
		cvtColor(newFrame, newGray_temp, CV_BGR2GRAY);
		warpPerspective(newGray_temp, newGray, perspective_matrix, aftersize);
		threshold(newGray, newGray_temp_temp, 175, 255, THRESH_TOZERO);

		/*Farneback光流法計算*/
		calcOpticalFlowFarneback(prevGray, newGray_temp_temp, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags);

		if (test == 1)
			drawOptFlowMap(flow, newGray, 20, CV_RGB(255, 0, 0));
		drawOptFlowMap(oldFlow, flow, newGray, 20, CV_RGB(255, 0, 0));

		/*儲存計算結果*/
		writer.write(newGray);

		//namedWindow("Output", WINDOW_NORMAL);
		//if (newGray.cols >= 1366 || newGray.rows >= 768)
		//	resizeWindow("Output", round(newGray.cols / 2), round(newGray.rows / 2));
		//imshow("Output", newGray);
		//waitKey(0);

		prevGray = newGray_temp_temp.clone();
		oldFlow = flow;

		////test
		//if (test == 100)
		//{
		//	system("pause");
		//}

	}

	timeEnd = clock();
	cout << "total time = " << (timeEnd - timeStart) / CLOCKS_PER_SEC << " s" << endl;
	system("pause");
	return 0;
}