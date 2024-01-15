// StereoDetection.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main()
{
	/*	----------双目相机基本参数----------	*/
	/*	可以通过MatLab相机标定工具箱得到	*/
	//所有类型必须为double-CV_64

	Mat inMatrixL = (Mat_<double>(3, 3) <<
		516.5066236, -1.444673028, 320.2950423,
		0, 516.5816117, 270.7881873,
		0.0, 0.0, 1.0);
	Mat inMatrixR = (Mat_<double>(3, 3) <<
		511.8428182, 1.295112628, 317.310253,
		0, 513.0748795, 269.5885026,
		0.0, 0.0, 1.0);
	Mat distCoeffsL = (Mat_<double>(1, 5)<<
		-0.046645194, 0.077595167, 0.012476819, -0.000711358, 0);
	Mat distCoeffsR = (Mat_<double>(1, 5) <<
		-0.061588946, 0.122384376, 0.011081232, -0.000750439, 0);;
	
	/*
	* rotation 和 translation
	* 为第二个摄像头到第一个摄像头的旋转矩阵和平移向量
	* 可以由MATLAB小工具得到，
	* 也可以由stereoCalibrate()函数得到
	* 
	*/
	Mat rotation = (Mat_<double>(3, 3) <<
		0.999911333, -0.004351508, 0.012585312,
		0.004184066, 0.999902792, 0.013300386,
		-0.012641965, -0.013246549, 0.999832341);
	Mat translation = (Mat_<double>(3, 1)<<
		-120.3559901, -0.188953775, -0.662073075);

	Size videoSize(640, 480);

	/*	----------双目相机立体矫正----------	*/
	
	//矫正旋转矩阵与投影矩阵
	//将相机坐标系下未矫正的点变换到其矫正坐标系下
	Mat rectifyRotationL;
	Mat rectifyRotationR;
	Mat rectifyProjectionL;
	Mat rectifyProjectionR;
	Mat Q;	//视差深度映射矩阵

	Mat rectifyMapL1;
	Mat rectifyMapL2;
	Mat rectifyMapR1;
	Mat rectifyMapR2;
	
	//立体校正——获取每个摄像头的映射矩阵
	stereoRectify(
		inMatrixL, distCoeffsL,
		inMatrixR, distCoeffsR,
		videoSize, rotation, translation,
		rectifyRotationL, rectifyRotationR,
		rectifyProjectionL, rectifyProjectionR,
		Q);


	//计算立体校正映射表
	initUndistortRectifyMap(
		inMatrixL, distCoeffsL,
		rectifyRotationL, rectifyProjectionL,
		videoSize, CV_16SC2,
		rectifyMapL1, rectifyMapL2);

	initUndistortRectifyMap(
		inMatrixR, distCoeffsR,
		rectifyRotationR, rectifyProjectionR,
		videoSize, CV_16SC2,
		rectifyMapR1, rectifyMapR2);

	//输出Q大矩阵！
	cout << Q << endl;


	VideoCapture v;
	v.open("car.avi");
	Mat tmp;
	Mat framL;
	Mat framR;
	Mat outPut;
	Mat disparity;

	/*	----------SGBM立体匹配----------	*/
	/*	
	*	blockSize         深度图成块，blocksize越低，其深度图就越零碎，0<blockSize<10
    *   
	*	img_channels      BGR图像的颜色通道，img_channels=3，不可更改
    *   
	*	numDisparities    SGBM感知的范围，越大生成的精度越好，速度越慢，需要被16整除，
	*					  如numDisparities取16、32、48、64等
    *                     
    *   mode              sgbm算法选择模式
	*					  以速度由快到慢为：STEREO_SGBM_MODE_SGBM_3WAY、STEREO_SGBM_MODE_HH4、STEREO_SGBM_MODE_SGBM、STEREO_SGBM_MODE_HH。精度反之
    * 
	*/
	Ptr<StereoSGBM> SGBM;
	//SGBM->
	//SGBM->create();

	while (true)
	{
		if (!v.read(tmp))
		{
			break;
		}
		//浅拷贝分割
		framL = tmp(Rect(0, 0, 640, 480));
		framR = tmp(Rect(640, 0, 640, 480));
		
		//这里发现了cvtColor可以深拷贝
		cvtColor(framL, framL, COLOR_BGR2GRAY);
		cvtColor(framR, framR, COLOR_BGR2GRAY);
	
		//立体校正
		remap(framL, framL, rectifyMapL1, rectifyMapL2, INTER_LINEAR);
		remap(framR, framR, rectifyMapR1, rectifyMapR2, INTER_LINEAR);
		
		SGBM->compute(framL, framR, disparity);

		hconcat(framL, framR, outPut);
		normalize(disparity, disparity, 0, 255, NORM_MINMAX);

		//cvtColor(outPut, outPut, COLOR_GRAY2BGR);
		
		// 绘制平行线
		for (int i = 1, iend = 15; i < iend; i++) {
			int h = outPut.rows / iend * i;
			line(outPut, Point2i(0, h), Point2i(outPut.cols, h), Scalar(0, 0, 255));
			line(tmp, Point2i(0, h), Point2i(outPut.cols, h), Scalar(0, 0, 255));
		}
		
		imshow("LeftCam", framL);
		imshow("ReftCam", framR);
		imshow("outPut", outPut);
		imshow("inPut", tmp);

		//waitKey(1000/20);

		//waitKey(0);
	}

	
	
	
	waitKey(0);
}
