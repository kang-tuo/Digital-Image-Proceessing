//============================================================================
// Name        : Dip4.cpp
// Author      : Ronny Haensch
// Version     : 2.0
// Copyright   : -
// Description : 
//============================================================================

#include "Dip4.h"

// Performes a circular shift in (dx,dy) direction
/*
in       :  input matrix
dx       :  shift in x-direction
dy       :  shift in y-direction
return   :  circular shifted matrix
*/
Mat Dip4::circShift(const Mat& in, int dx, int dy){
	// TO DO !!!
    
    Mat circShift_Mat = Mat::zeros(in.rows, in.cols, in.type());
    
    for (int ix = 0; ix < in.rows; ++ix)
    {
        int ix_cs = (ix + dx) % in.rows;
        if (ix_cs < 0)
        {
            ix_cs = in.rows + ix_cs;
        }
        
        for (int iy = 0; iy < in.cols; ++iy)
        {
            int iy_cs = (iy + dy) % in.cols;
            if (iy_cs < 0)
            {
                iy_cs = in.rows + iy_cs;
            }
            
            circShift_Mat.at<float>(iy_cs, ix_cs) = in.at<float>(iy, ix);
            
        }
    }
    
    return circShift_Mat;
}

// Function applies the inverse filter to restorate a degraded image
/*
degraded :  degraded input image
filter   :  filter which caused degradation
return   :  restorated output image
*/
Mat Dip4::inverseFilter(const Mat& degraded, const Mat& filter){
	// TO DO !!!

    Mat kernelExp = Mat::zeros(degraded.size(), degraded.type());
    for (int x = 0; x < filter.cols; x++)
    {
        for (int y = 0; y < filter.rows; y++)
        {
            kernelExp.at<float>(y, x) = filter.at<float>(y, x);
        }
    }
    Mat kernelExpStfd = circShift(kernelExp, -(filter.cols / 2), -(filter.rows / 2));
    
    Mat complex[] = {kernelExpStfd, Mat::zeros(degraded.size(), degraded.type())};
    Mat dftcomplex;
    merge(complex, 2, dftcomplex);
    
    dft(dftcomplex, dftcomplex);
    split(dftcomplex, complex);
    
    float max = 0;
    Mat mag = Mat(degraded.size(), degraded.type());
    for(int i = 0;i < degraded.rows;i++)
        for(int j = 0;j < degraded.cols;j++)
        {
            mag.at<float>(i,j) = sqrt(pow(complex[0].at<float>(i,j),2) + pow(complex[1].at<float>(i,j),2));
            if(mag.at<float>(i,j) > max)
                max = mag.at<float>(i,j);
        }
    
    for(int i = 0;i < degraded.rows;i++)
        for(int j = 0;j < degraded.cols;j++)
        {
            if(mag.at<float>(i,j) < (0.05 * max))// 1/T
            {
                complex[0].at<float>(i,j) = (0.05 * max);
                complex[1].at<float>(i,j) = 0;
            }
            else// 1/P
            {
                complex[0].at<float>(i,j) /= pow(mag.at<float>(i,j), 2);
                complex[1].at<float>(i,j) = -complex[1].at<float>(i,j)/ pow(mag.at<float>(i,j), 2);
            }
        }
    
    Mat invfilter = Mat(degraded.size(), degraded.type());
    merge(complex, 2, invfilter);
    
    Mat complex2[] = {degraded, Mat::zeros(degraded.size(), degraded.type())};
    Mat dftcomplex2;
    merge(complex2, 2, dftcomplex2);
    
    dft(dftcomplex2, dftcomplex2);
    
    Mat finalImg = Mat(degraded.size(), degraded.type());
    mulSpectrums(dftcomplex2, invfilter, dftcomplex2, 0);

    dft(dftcomplex2, dftcomplex2, DFT_INVERSE + DFT_SCALE);
    split(dftcomplex2, complex);

    for(int i = 0;i < degraded.rows;i++)
        for(int j = 0;j < degraded.cols;j++)
        {
            finalImg.at<float>(i,j) = complex[0].at<float>(i,j);// sqrt(pow(complex[0].at<float>(i,j), 2) + pow(complex[1].at<float>(i,j), 2));
     //don't know which to use
        }
    threshold(finalImg, finalImg, 255, 255, THRESH_TRUNC);
    return finalImg;
}

// Function applies the wiener filter to restorate a degraded image
/*
degraded :  degraded input image
filter   :  filter which caused degradation
snr      :  signal to noise ratio of the input image
return   :   restorated output image
*/
Mat Dip4::wienerFilter(const Mat& degraded, const Mat& filter, double snr){
	// TO DO !!!
    
    Mat kernelExp = Mat::zeros(degraded.size(), degraded.type());
    for (int x = 0; x < filter.cols; x++)
    {
        for (int y = 0; y < filter.rows; y++)
        {
            kernelExp.at<float>(y, x) = filter.at<float>(y, x);
        }
    }
    Mat kernelExpStfd = circShift(kernelExp, -(filter.cols / 2), -(filter.rows / 2));
    
    Mat complex[] = {kernelExpStfd, Mat::zeros(degraded.size(), degraded.type())};
    Mat dftcomplex;
    merge(complex, 2, dftcomplex);
    
    dft(dftcomplex, dftcomplex);
    split(dftcomplex, complex);
    
    Mat mag = Mat(degraded.size(), degraded.type());
    for(int i = 0;i < degraded.rows;i++)
        for(int j = 0;j < degraded.cols;j++)
        {
            mag.at<float>(i,j) = sqrt(pow(complex[0].at<float>(i,j),2) + pow(complex[1].at<float>(i,j),2));
        }
    
    for(int i = 0;i < degraded.rows;i++)
        for(int j = 0;j < degraded.cols;j++)
        {
            complex[0].at<float>(i,j) /= pow(mag.at<float>(i,j), 2) + 1/(snr * snr);
            complex[1].at<float>(i,j) = -complex[1].at<float>(i,j) /
            (pow(mag.at<float>(i,j), 2) + 1/(snr * snr));
        }
    
    Mat invfilter = Mat(degraded.size(), degraded.type());
    merge(complex, 2, invfilter);
    
    Mat complex2[] = {degraded, Mat::zeros(degraded.size(), degraded.type())};
    Mat dftcomplex2;
    merge(complex2, 2, dftcomplex2);
    
    dft(dftcomplex2, dftcomplex2);
    
    Mat finalImg = Mat(degraded.size(), degraded.type());
    mulSpectrums(dftcomplex2, invfilter, dftcomplex2, 0);
    
    dft(dftcomplex2, dftcomplex2, DFT_INVERSE + DFT_SCALE);
    split(dftcomplex2, complex);
    for(int i = 0;i < degraded.rows;i++)
        for(int j = 0;j < degraded.cols;j++)
        {
            finalImg.at<float>(i,j) = complex[0].at<float>(i,j);// sqrt(pow(complex[0].at<float>(i,j), 2) + pow(complex[1].at<float>(i,j), 2));
            //don't know which to use
        }
    threshold(finalImg, finalImg, 255, 255, THRESH_TRUNC);
    return finalImg;
}

/* *****************************
  GIVEN FUNCTIONS
***************************** */

// function calls processing function
/*
in                   :  input image
restorationType     :  integer defining which restoration function is used
kernel               :  kernel used during restoration
snr                  :  signal-to-noise ratio (only used by wieder filter)
return               :  restorated image
*/
Mat Dip4::run(const Mat& in, string restorationType, const Mat& kernel, double snr){

   if (restorationType.compare("wiener")==0){
      return wienerFilter(in, kernel, snr);
   }else{
      return inverseFilter(in, kernel);
   }

}

// function degrades the given image with gaussian blur and additive gaussian noise
/*
img         :  input image
degradedImg :  degraded output image
filterDev   :  standard deviation of kernel for gaussian blur
snr         :  signal to noise ratio for additive gaussian noise
return      :  the used gaussian kernel
*/
Mat Dip4::degradeImage(const Mat& img, Mat& degradedImg, double filterDev, double snr){

    int kSize = round(filterDev*3)*2 - 1;
   
    Mat gaussKernel = getGaussianKernel(kSize, filterDev, CV_32FC1);
    gaussKernel = gaussKernel * gaussKernel.t();

    Mat imgs = img.clone();
    dft( imgs, imgs, CV_DXT_FORWARD, img.rows);
    Mat kernels = Mat::zeros( img.rows, img.cols, CV_32FC1);
    int dx, dy; dx = dy = (kSize-1)/2.;
    for(int i=0; i<kSize; i++) for(int j=0; j<kSize; j++) kernels.at<float>((i - dy + img.rows) % img.rows,(j - dx + img.cols) % img.cols) = gaussKernel.at<float>(i,j);
	dft( kernels, kernels, CV_DXT_FORWARD );
	mulSpectrums( imgs, kernels, imgs, 0 );
	dft( imgs, degradedImg, CV_DXT_INV_SCALE, img.rows );
	
    Mat mean, stddev;
    meanStdDev(img, mean, stddev);

    Mat noise = Mat::zeros(img.rows, img.cols, CV_32FC1);
    randn(noise, 0, stddev.at<double>(0)/snr);
    degradedImg = degradedImg + noise;
    threshold(degradedImg, degradedImg, 255, 255, CV_THRESH_TRUNC);
    threshold(degradedImg, degradedImg, 0, 0, CV_THRESH_TOZERO);

    return gaussKernel;
}

// Function displays image (after proper normalization)
/*
win   :  Window name
img   :  image that shall be displayed
cut   :  determines whether to cut or scale values outside of [0,255] range
*/
void Dip4::showImage(const char* win, const Mat& img, bool cut){

   Mat tmp = img.clone();

   if (tmp.channels() == 1){
      if (cut){
         threshold(tmp, tmp, 255, 255, CV_THRESH_TRUNC);
         threshold(tmp, tmp, 0, 0, CV_THRESH_TOZERO);
      }else
         normalize(tmp, tmp, 0, 255, CV_MINMAX);
         
      tmp.convertTo(tmp, CV_8UC1);
   }else{
      tmp.convertTo(tmp, CV_8UC3);
   }
   imshow(win, tmp);
}

// function calls basic testing routines to test individual functions for correctness
void Dip4::test(void){

   test_circShift();
   cout << "Press enter to continue"  << endl;
   cin.get();

}

void Dip4::test_circShift(void){
   
   Mat in = Mat::zeros(3,3,CV_32FC1);
   in.at<float>(0,0) = 1;
   in.at<float>(0,1) = 2;
   in.at<float>(1,0) = 3;
   in.at<float>(1,1) = 4;
   Mat ref = Mat::zeros(3,3,CV_32FC1);
   ref.at<float>(0,0) = 4;
   ref.at<float>(0,2) = 3;
   ref.at<float>(2,0) = 2;
   ref.at<float>(2,2) = 1;
   
   if (sum((circShift(in, -1, -1) == ref)).val[0]/255 != 9){
      cout << "ERROR: Dip4::circShift(): Result of circshift seems to be wrong!" << endl;
      return;
   }
   cout << "Message: Dip4::circShift() seems to be correct" << endl;
}
