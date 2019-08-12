//============================================================================
// Name        : Dip5.cpp
// Author      : Ronny Haensch
// Version     : 2.0
// Copyright   : -
// Description : 
//============================================================================

#include "Dip5.h"

// uses structure tensor to define interest points (foerstner)
void Dip5::getInterestPoints(const Mat& img, double sigma, vector<KeyPoint>& points){
	// TO DO !!!
	int kSize = 3;
	Mat fdkx = createFstDevKernel(sigma);
	Mat fdky = fdkx.t();
	Mat gx,gy;
	filter2D(img, gx, CV_32F, fdkx);
	filter2D(img, gy, CV_32F, fdky);

	//average
	Mat avgxx = AvgGWindow(gx, gx);
	Mat avgyy = AvgGWindow(gy, gy);
	Mat avgxy = AvgGWindow(gx, gy);
	Mat avgyx = AvgGWindow(gy, gx);
	
	
	//trace
	Mat tr =Mat::zeros(gx.rows, gx.cols, CV_32F);	
	for (int i = 0; i < gx.rows; i++)
	{
		for (int j = 0; j < gx.cols; j++)
		{
			tr.at<float>(i, j) = avgxx.at<float>(i, j) + avgyy.at<float>(i, j);
		}
			
	}
	

	//determinate
	Mat det = Mat::zeros(gx.rows, gx.cols, CV_32F);
	for (int i = 0; i < gx.rows; i++)
	{
		for (int j = 0; j < gx.cols; j++)
		{
			det.at<float>(i, j) = avgxx.at<float>(i, j) * avgxx.at<float>(i, j)- avgxy.at<float>(i, j) * avgyx.at<float>(i, j);
		}

	}


	//weight
    Mat w= Mat::zeros(tr.rows, tr.cols, CV_32F);
	for (int i = 0; i < tr.rows; i++)
	{
		for (int j = 0; j < tr.cols; j++)
		{
			w.at<float>(i, j) = det.at<float>(i, j) / tr.at<float>(i, j);
		}

	}
    nonMaxSuppression(w);
	float wsum = 0.0;
	for (int i = kSize; i < w.rows-kSize; i++)
	{
		for (int j = kSize; j < w.cols-kSize; j++)
		{
			wsum = wsum + w.at<float>(i, j);
		}
	}
	float wmean = wsum / (w.rows*w.cols);
	threshold(w, w, 255, 1.5*wmean, THRESH_BINARY);


	//isotropy 
	Mat q = Mat::zeros(tr.rows, tr.cols, CV_32F);
	for (int i = 0; i < tr.rows; i++)
	{
		for (int j = 0; j < tr.cols; j++)
		{
			q.at<float>(i, j) = 4*det.at<float>(i, j) /( tr.at<float>(i, j)*tr.at<float>(i, j));
		}

	}
	nonMaxSuppression(q);
	threshold(q, q, 255, 0.75, THRESH_BINARY);
	
	
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++) 
		{
					if (w.at<float>(i,j))
					{
						if (q.at<float>(i, j))
						{
							points.push_back(KeyPoint(j, i, 3));
						}
					}	    
		}
	}
		
	/*showImage(w, "weights", 0, true, true);
	showImage(q, "isotropy", 0, true, true);*/
					
}





Mat Dip5::AvgGWindow(Mat g1, Mat g2) {
	
	int g1_row = g1.rows;
	int g1_col = g1.cols;
	int kSize = 3;

	Mat avg1 = Mat::zeros(g1_row, g1_col, CV_32F);				
	for (int i = 0; i <= g1_row - kSize; i++)
	{
		for (int j = 0; j <= g1_col - kSize; j++)
		{
			float anchor = 0.0;
			for (int k = 0; k < kSize; k++)
			{
				for (int l = 0; l < kSize; l++)
				{
					anchor += (g1.at<float>(i + k, j + l) * g2.at<float>(kSize - k - 1, kSize - l - 1));
				}

			}
			avg1.at<float>(i + kSize / 2, j + kSize / 2) = anchor;
		}
	}

	Mat avg2 = Mat::zeros(g1.rows, g1.cols, CV_32F);
	GaussianBlur(avg1, avg2, Size(kSize, kSize), sigma);
	return avg2;

}

// creates kernel representing fst derivative of a Gaussian kernel in x-direction
/*
sigma	standard deviation of the Gaussian kernel
return	the calculated kernel
*/
Mat Dip5::createFstDevKernel(double sigma){
	//// TO DO !!!
	sigma = this->sigma;
    int kSize =round(sigma * 3) * 2 - 1;
	Mat gkx = getGaussianKernel(kSize, sigma, CV_32F);
	Mat gky = gkx;
	Mat gk = gkx * gky.t();
	Mat fdkx = Mat::zeros(kSize, kSize, CV_32F);
	for (int i = 0; i < kSize; i++) {
		for (int j = 0; j < kSize; j++) {
			fdkx.at<float>(i, j) = ( -(i-1)/(sigma*sigma)) * gk.at<float>(i, j);
		}
	}
	return fdkx;
	
}

/* *****************************
  GIVEN FUNCTIONS
***************************** */

// function calls processing function
/*
in		:  input image
points	:	detected keypoints
*/
void Dip5::run(const Mat& in, vector<KeyPoint>& points){
   this->getInterestPoints(in, this->sigma, points);
}

// non-maxima suppression
// if any of the pixel at the 4-neighborhood is greater than current pixel, set it to zero
Mat Dip5::nonMaxSuppression(const Mat& img){

	Mat out = img.clone();
	
	for(int x=1; x<out.cols-1; x++){
		for(int y=1; y<out.rows-1; y++){
			if ( img.at<float>(y-1, x) >= img.at<float>(y, x) ){
				out.at<float>(y, x) = 0;
				continue;
			}
			if ( img.at<float>(y, x-1) >= img.at<float>(y, x) ){
				out.at<float>(y, x) = 0;
				continue;
			}
			if ( img.at<float>(y, x+1) >= img.at<float>(y, x) ){
				out.at<float>(y, x) = 0;
				continue;
			}
			if ( img.at<float>( y+1, x) >= img.at<float>(y, x) ){
				out.at<float>(y, x) = 0;
				continue;
			}
		}
	}
	return out;
}

// Function displays image (after proper normalization)
/*
win   :  Window name
img   :  Image that shall be displayed
cut   :  whether to cut or scale values outside of [0,255] range
*/
void Dip5::showImage(const Mat& img, const char* win, int wait, bool show, bool save){
  
    Mat aux = img.clone();

    // scale and convert
    if (img.channels() == 1)
		normalize(aux, aux, 0, 255, CV_MINMAX);
		aux.convertTo(aux, CV_8UC1);
    // show
    if (show){
      imshow( win, aux);
      waitKey(wait);
    }
    // save
    if (save)
      imwrite( (string(win)+string(".png")).c_str(), aux);
}
