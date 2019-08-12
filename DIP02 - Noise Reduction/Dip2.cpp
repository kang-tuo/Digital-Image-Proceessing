//============================================================================
// Name        : Dip2.cpp
// Author      : Ronny Haensch
// Version     : 2.0
// Copyright   : -
// Description : 
//============================================================================
#include "Dip2.h"
#include <math.h>
// convolution in spatial domain
/*
src:     input image
kernel:  filter kernel
return:  convolution result
*/
Mat Dip2::spatialConvolution(Mat& src, Mat& kernel) {
	// TO DO !!
	int oi_row = src.rows;
	int oi_col = src.cols;
	int k_row = kernel.rows;
	int k_col = kernel.cols;

	Mat pic = Mat::zeros(oi_row, oi_col, CV_32F);
	for (int i = 0; i <= oi_row - k_row; i++)
	{
		for (int j = 0; j <= oi_col - k_col; j++)
		{
			float anchor = 0.0;
			for (int k = 0; k < k_row; k++)
			{
				for (int l = 0; l < k_col; l++)
				{
					anchor +=( src.at<float>(i + k, j + l) * kernel.at<float>(k_row - k - 1, k_col - l - 1));
				}

			}
			pic.at<float>(i + k_row / 2, j + k_col / 2) = anchor;
		}
	}

	return pic.clone();

}

// the average filter
// HINT: you might want to use Dip2::spatialConvolution(...) within this function
/*
src:     input image
kSize:   window size used by local average
return:  filtered image
*/
Mat Dip2::averageFilter(Mat& src, int kSize) {
	// TO DO !!
	Mat kernel = Mat::ones(kSize, kSize, CV_32F) * (1.0 / (kSize*kSize));

	src = spatialConvolution(src, kernel);
	return src.clone();
}

// the median filter
/*
src:     input image
kSize:   window size used by median operation
return:  filtered image
*/
Mat Dip2::medianFilter(Mat& src, int kSize) {
	// TO DO !!
	//Mat pic = Mat::zeros(src.rows, src.cols, CV_32F);
	float* array = new float[kSize * kSize];
	for (int i = kSize / 2; i < src.rows - kSize / 2; i++)
		for (int j = kSize / 2; j < src.cols - kSize / 2; j++)
		{
			for (int k = 0; k < kSize; k++)
				for (int l = 0; l < kSize; l++)
					array[k * kSize + l] = src.at<float>(i - kSize / 2 + k, j - kSize / 2 + l);
			std::sort(array, array + kSize * kSize);
			src.at<float>(i, j) = array[kSize * kSize / 2];
		}

		return src.clone();
}

// the bilateral filter
/*
src:     input image
kSize:   size of the kernel --> used to compute std-dev of spatial kernel
sigma:   standard-deviation of the radiometric kernel
return:  filtered image
*/
Mat Dip2::bilateralFilter(Mat& src, int kSize, double sigma) {


	double sigma_1 = 0.3*((kSize - 1)*0.5 - 1) + 0.8;
	double u = (kSize - 1) / 2;

	Mat pic = Mat::zeros(src.rows, src.cols, CV_32F);
	for (int i = kSize / 2; i < src.rows - kSize / 2; i++)
		for (int j = kSize / 2; j < src.cols - kSize / 2; j++)
		{
			double anchor = 0.0;
			double sum = 0;
			for (int k = 0; k < kSize; k++)
				for (int l = 0; l < kSize; l++)
				{
					double h_s = (1 / (2 * 3.1415926*sigma_1*sigma_1))*exp(-((pow((k - u), 2) + pow((l - u), 2)) / (2 * sigma_1*sigma_1)));
					double h_r = (1 / (2 * 3.1415926*sigma*sigma))*exp(-(pow(((src.at<float>((i + k - u), (j + l - u))) - (src.at<float>(i, j))), 2)) / (2 * sigma*sigma));
					double weight = h_s*h_r;
					anchor += (weight * src.at<float>(i + k - u, j + l - u));
					sum += weight;
				}

			anchor = anchor * (1 / sum);
			pic.at<float>(i, j) = anchor;
		}



	return pic.clone();

}

// the non-local means filter
/*
src:   		input image
searchSize: size of search region
sigma: 		Optional parameter for weighting function
return:  	filtered image
*/
Mat Dip2::nlmFilter(Mat& src, int searchSize, double sigma) {

	return src.clone();

}

/* *****************************
GIVEN FUNCTIONS
***************************** */

// function loads input image, calls processing function, and saves result
void Dip2::run(void) {

	// load images as grayscale
	cout << "load images" << endl;
	Mat noise1 = imread("noiseType_1.jpg", 0);
	if (!noise1.data) {
		cout << "noiseType_1.jpg not found" << endl;
		cout << "Press enter to exit" << endl;
		cin.get();
		exit(-3);
	}
	noise1.convertTo(noise1, CV_32FC1);
	Mat noise2 = imread("noiseType_2.jpg", 0);
	if (!noise2.data) {
		cout << "noiseType_2.jpg not found" << endl;
		cout << "Press enter to exit" << endl;
		cin.get();
		exit(-3);
	}
	noise2.convertTo(noise2, CV_32FC1);
	cout << "done" << endl;

	// apply noise reduction
	// TO DO !!!
	// ==> Choose appropriate noise reduction technique with appropriate parameters
	// ==> "average" or "median"? Why?

	// ==> try also "bilateral" (and if implemented "nlm")
	cout << "reduce noise" << endl;
	Mat restorated1 = noiseReduction(noise1, "median", 7);
	Mat restorated2 = noiseReduction(noise2, "average", 7);
	Mat restorated3 = noiseReduction(noise2, "bilateral", 7, 0.8);

	cout << "done" << endl;

	// save images
	cout << "save results" << endl;
	imwrite("restorated1.jpg", restorated1);
	imwrite("restorated2.jpg", restorated2);
	imwrite("restorated3.jpg", restorated3);

	cout << "done" << endl;

}

// noise reduction
/*
src:     input image
method:  name of noise reduction method that shall be performed
"average" ==> moving average
"median" ==> median filter
"bilateral" ==> bilateral filter
"nlm" ==> non-local means filter
kSize:   (spatial) kernel size
param:   if method == "bilateral", standard-deviation of radiometric kernel; if method == "nlm", (optional) parameter for similarity function
can be ignored otherwise (default value = 0)
return:  output image
*/
Mat Dip2::noiseReduction(Mat& src, string method, int kSize, double param) {

	// apply moving average filter
	if (method.compare("average") == 0) {
		return averageFilter(src, kSize);
	}
	// apply median filter
	if (method.compare("median") == 0) {
		return medianFilter(src, kSize);
	}
	// apply bilateral filter
	if (method.compare("bilateral") == 0) {
		return bilateralFilter(src, kSize, param);
	}
	// apply adaptive average filter
	if (method.compare("nlm") == 0) {
		return nlmFilter(src, kSize, param);
	}

	// if none of above, throw warning and return copy of original
	cout << "WARNING: Unknown filtering method! Returning original" << endl;
	cout << "Press enter to continue" << endl;
	cin.get();
	return src.clone();

}

// generates and saves different noisy versions of input image
/*
fname:   path to the input image
*/
void Dip2::generateNoisyImages(string fname) {

	// load image, force gray-scale
	cout << "load original image" << endl;
	Mat img = imread(fname, 0);
	if (!img.data) {
		cout << "ERROR: file " << fname << " not found" << endl;
		cout << "Press enter to exit" << endl;
		cin.get();
		exit(-3);
	}

	// convert to floating point precision
	img.convertTo(img, CV_32FC1);
	cout << "done" << endl;

	// save original
	imwrite("original.jpg", img);

	// generate images with different types of noise
	cout << "generate noisy images" << endl;

	// some temporary images
	Mat tmp1(img.rows, img.cols, CV_32FC1);
	Mat tmp2(img.rows, img.cols, CV_32FC1);
	// first noise operation
	float noiseLevel = 0.15;
	randu(tmp1, 0, 1);
	threshold(tmp1, tmp2, noiseLevel, 1, CV_THRESH_BINARY);
	multiply(tmp2, img, tmp2);
	threshold(tmp1, tmp1, 1 - noiseLevel, 1, CV_THRESH_BINARY);
	tmp1 *= 255;
	tmp1 = tmp2 + tmp1;
	threshold(tmp1, tmp1, 255, 255, CV_THRESH_TRUNC);
	// save image
	imwrite("noiseType_1.jpg", tmp1);

	// second noise operation
	noiseLevel = 50;
	randn(tmp1, 0, noiseLevel);
	tmp1 = img + tmp1;
	threshold(tmp1, tmp1, 255, 255, CV_THRESH_TRUNC);
	threshold(tmp1, tmp1, 0, 0, CV_THRESH_TOZERO);
	// save image
	imwrite("noiseType_2.jpg", tmp1);

	cout << "done" << endl;
	cout << "Please run now: dip2 restorate" << endl;

}

// function calls some basic testing routines to test individual functions for correctness
void Dip2::test(void) {

	test_spatialConvolution();
	test_averageFilter();
	test_medianFilter();

	cout << "Press enter to continue" << endl;
	cin.get();

}

// checks basic properties of the convolution result
void Dip2::test_spatialConvolution(void) {

	Mat input = Mat::ones(9, 9, CV_32FC1);
	input.at<float>(4, 4) = 255;
	Mat kernel = Mat(3, 3, CV_32FC1, 1. / 9.);

	Mat output = spatialConvolution(input, kernel);


	if ((input.cols != output.cols) || (input.rows != output.rows)) {
		cout << "ERROR: Dip2::spatialConvolution(): input.size != output.size --> Wrong border handling?" << endl;
		return;
	}
	if ((sum(output.row(0) < 0).val[0] > 0) ||
		(sum(output.row(0) > 255).val[0] > 0) ||
		(sum(output.row(8) < 0).val[0] > 0) ||
		(sum(output.row(8) > 255).val[0] > 0) ||
		(sum(output.col(0) < 0).val[0] > 0) ||
		(sum(output.col(0) > 255).val[0] > 0) ||
		(sum(output.col(8) < 0).val[0] > 0) ||
		(sum(output.col(8) > 255).val[0] > 0)) {
		cout << "ERROR: Dip2::spatialConvolution(): Border of convolution result contains too large/small values --> Wrong border handling?" << endl;
		return;
	}
	else {
		if ((sum(output < 0).val[0] > 0) ||
			(sum(output > 255).val[0] > 0)) {
			cout << "ERROR: Dip2::spatialConvolution(): Convolution result contains too large/small values!" << endl;
			return;
		}
	}
	float ref[9][9] = { { 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	{ 0, 1, 1, 1, 1, 1, 1, 1, 0 },
	{ 0, 1, 1, 1, 1, 1, 1, 1, 0 },
	{ 0, 1, 1, (8 + 255) / 9., (8 + 255) / 9., (8 + 255) / 9., 1, 1, 0 },
	{ 0, 1, 1, (8 + 255) / 9., (8 + 255) / 9., (8 + 255) / 9., 1, 1, 0 },
	{ 0, 1, 1, (8 + 255) / 9., (8 + 255) / 9., (8 + 255) / 9., 1, 1, 0 },
	{ 0, 1, 1, 1, 1, 1, 1, 1, 0 },
	{ 0, 1, 1, 1, 1, 1, 1, 1, 0 },
	{ 0, 0, 0, 0, 0, 0, 0, 0, 0 } };
	for (int y = 1; y<8; y++) {
		for (int x = 1; x<8; x++) {
			if (abs(output.at<float>(y, x) - ref[y][x]) > 0.0001) {
				cout << "ERROR: Dip2::spatialConvolution(): Convolution result contains wrong values!" << endl;
				return;
			}
		}
	}
	input.setTo(0);
	input.at<float>(4, 4) = 255;
	kernel.setTo(0);
	kernel.at<float>(0, 0) = -1;
	output = spatialConvolution(input, kernel);
	if (abs(output.at<float>(5, 5) + 255.) < 0.0001) {
		cout << "ERROR: Dip2::spatialConvolution(): Is filter kernel \"flipped\" during convolution? (Check lecture/exercise slides)" << endl;
		return;
	}
	if ((abs(output.at<float>(2, 2) + 255.) < 0.0001) || (abs(output.at<float>(4, 4) + 255.) < 0.0001)) {
		cout << "ERROR: Dip2::spatialConvolution(): Is anchor point of convolution the centre of the filter kernel? (Check lecture/exercise slides)" << endl;
		return;
	}
	cout << "Message: Dip2::spatialConvolution() seems to be correct" << endl;
}

// checks basic properties of the filtering result
void Dip2::test_averageFilter(void) {

	Mat input = Mat::ones(9, 9, CV_32FC1);
	input.at<float>(4, 4) = 255;

	Mat output = averageFilter(input, 3);

	if ((input.cols != output.cols) || (input.rows != output.rows)) {
		cout << "ERROR: Dip2::averageFilter(): input.size != output.size --> Wrong border handling?" << endl;
		return;
	}
	if ((sum(output.row(0) < 0).val[0] > 0) ||
		(sum(output.row(0) > 255).val[0] > 0) ||
		(sum(output.row(8) < 0).val[0] > 0) ||
		(sum(output.row(8) > 255).val[0] > 0) ||
		(sum(output.col(0) < 0).val[0] > 0) ||
		(sum(output.col(0) > 255).val[0] > 0) ||
		(sum(output.col(8) < 0).val[0] > 0) ||
		(sum(output.col(8) > 255).val[0] > 0)) {
		cout << "ERROR: Dip2::averageFilter(): Border of result contains too large/small values --> Wrong border handling?" << endl;
		return;
	}
	else {
		if ((sum(output < 0).val[0] > 0) ||
			(sum(output > 255).val[0] > 0)) {
			cout << "ERROR: Dip2::averageFilter(): Result contains too large/small values!" << endl;
			return;
		}
	}
	float ref[9][9] = { { 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	{ 0, 1, 1, 1, 1, 1, 1, 1, 0 },
	{ 0, 1, 1, 1, 1, 1, 1, 1, 0 },
	{ 0, 1, 1, (8 + 255) / 9., (8 + 255) / 9., (8 + 255) / 9., 1, 1, 0 },
	{ 0, 1, 1, (8 + 255) / 9., (8 + 255) / 9., (8 + 255) / 9., 1, 1, 0 },
	{ 0, 1, 1, (8 + 255) / 9., (8 + 255) / 9., (8 + 255) / 9., 1, 1, 0 },
	{ 0, 1, 1, 1, 1, 1, 1, 1, 0 },
	{ 0, 1, 1, 1, 1, 1, 1, 1, 0 },
	{ 0, 0, 0, 0, 0, 0, 0, 0, 0 } };
	for (int y = 1; y<8; y++) {
		for (int x = 1; x<8; x++) {
			if (abs(output.at<float>(y, x) - ref[y][x]) > 0.0001) {
				cout << "ERROR: Dip2::averageFilter(): Result contains wrong values!" << endl;
				return;
			}
		}
	}
	cout << "Message: Dip2::averageFilter() seems to be correct" << endl;
}

// checks basic properties of the filtering result
void Dip2::test_medianFilter(void) {

	Mat input = Mat::ones(9, 9, CV_32FC1);
	input.at<float>(4, 4) = 255;

	Mat output = medianFilter(input, 3);

	if ((input.cols != output.cols) || (input.rows != output.rows)) {
		cout << "ERROR: Dip2::medianFilter(): input.size != output.size --> Wrong border handling?" << endl;
		return;
	}
	if ((sum(output.row(0) < 0).val[0] > 0) ||
		(sum(output.row(0) > 255).val[0] > 0) ||
		(sum(output.row(8) < 0).val[0] > 0) ||
		(sum(output.row(8) > 255).val[0] > 0) ||
		(sum(output.col(0) < 0).val[0] > 0) ||
		(sum(output.col(0) > 255).val[0] > 0) ||
		(sum(output.col(8) < 0).val[0] > 0) ||
		(sum(output.col(8) > 255).val[0] > 0)) {
		cout << "ERROR: Dip2::medianFilter(): Border of result contains too large/small values --> Wrong border handling?" << endl;
		return;
	}
	else {
		if ((sum(output < 0).val[0] > 0) ||
			(sum(output > 255).val[0] > 0)) {
			cout << "ERROR: Dip2::medianFilter(): Result contains too large/small values!" << endl;
			return;
		}
	}
	for (int y = 1; y<8; y++) {
		for (int x = 1; x<8; x++) {
			if (abs(output.at<float>(y, x) - 1.) > 0.0001) {
				cout << "ERROR: Dip2::medianFilter(): Result contains wrong values!" << endl;
				return;
			}
		}
	}
	cout << "Message: Dip2::medianFilter() seems to be correct" << endl;

}
