// StereoDetection.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void mouse(int event, int x, int y, int flags, void*);

Mat threeD;


int main()
{
	/*	----------双目相机基本参数----------	*/
	/*	可以通过MatLab相机标定工具箱得到	*/
	//所有类型必须为double-CV_64

	Mat inMatrixL = (Mat_<double>(3, 3) <<
		508.025383095559, -4.76577646636038, 298.758408695013,
		0, 516.367682321281, 269.943747097106,
		0.0, 0.0, 1.0);
	Mat inMatrixR = (Mat_<double>(3, 3) <<
		508.010513516867, -4.25470144125126, 316.923841134557,
		0, 514.076781977890, 268.517991005795,
		0.0, 0.0, 1.0);
	Mat distCoeffsL = (Mat_<double>(1, 5)<<
		-0.0467613211617264, 0.00456959237953496, 0.00599468139779429, 0.00562291627888502, 0);
	Mat distCoeffsR = (Mat_<double>(1, 5) <<
		-0.0348340455225505, 0.0319726382145138, 0.00730631911282040, 0.00880828926688701, 0);;
	
	/*
	* rotation 和 translation
	* 为第二个摄像头到第一个摄像头的旋转矩阵和平移向量
	* 可以由MATLAB小工具得到，
	* 也可以由stereoCalibrate()函数得到
	* 
	*/
	Mat rotation = (Mat_<double>(3, 3) <<
		0.999385581703916, -0.000462918384616362, -0.0350463234718745,
		0.000595261372085020, 0.999992731840344, 0.00376589038394706,
		0.0350443254497070, -0.00378443827459052, 0.999378593567383);
	Mat translation = (Mat_<double>(3, 1)<<
		-61.4675664088485, 0.340648089329872, -3.97468175020695);

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
	//v.open("car.avi");
	v.open("demo.mp4");
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
	int blockSize = 5;
	int img_channels = 1;
	Ptr<StereoSGBM> SGBM = StereoSGBM::create();
	SGBM->setMinDisparity(1);
	SGBM->setBlockSize(blockSize);
	SGBM->setNumDisparities(64);
	SGBM->setBlockSize(blockSize);
	SGBM->setP1(8 * img_channels * blockSize * blockSize);
	SGBM->setP1(32 * img_channels * blockSize * blockSize);
	SGBM->setDisp12MaxDiff(-1);
	SGBM->setPreFilterCap(1);
	SGBM->setSpeckleRange(1);
	SGBM->setSpeckleWindowSize(100);
	SGBM->setUniquenessRatio(10);
	SGBM->setMode(StereoSGBM::MODE_HH);

	char q = '0';
	int flag = 0;

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
		
		if (q == 'q' || flag == 1)
		{
			flag = 1;
		SGBM->compute(framL, framR, disparity);
			reprojectImageTo3D(disparity, threeD, Q, true);
			normalize(disparity, disparity, 0, 255, NORM_MINMAX, CV_8U);
			threeD *= 16;
			imshow("disparity", disparity);
			setMouseCallback("disparity", mouse);

		}
		if (q == 'a')
		{
			flag = 0;
		}

		//hconcat(framL, framR, outPut);

		//cvtColor(outPut, outPut, COLOR_GRAY2BGR);
		
		//// 绘制平行线
		//for (int i = 1, iend = 15; i < iend; i++) {
		//	int h = outPut.rows / iend * i;
		//	line(outPut, Point2i(0, h), Point2i(outPut.cols, h), Scalar(0, 0, 255));
		//	line(tmp, Point2i(0, h), Point2i(outPut.cols, h), Scalar(0, 0, 255));
		//}
		
		//imshow("LeftCam", framL);
		//imshow("ReftCam", framR);
		//imshow("outPut", outPut);
		imshow("inPut", tmp);

		q = waitKey(1000/30);

		//waitKey(0);
	}

	
	
	
	waitKey(0);
}


void mouse(int event, int x, int y, int flags, void*)
{
	
	if (event == EVENT_LBUTTONDOWN)
	{
		cout << "pt鼠标坐标为：" << Point(x, y) << endl;

		Vec3f coord = threeD.at<Vec3f>(y, x);
		double dist = sqrt(coord[0] * coord[0] + coord[1] * coord[1] + coord[2] * coord[2]);
		
		cout << "距离为" << dist/1000 << endl;


	}


}