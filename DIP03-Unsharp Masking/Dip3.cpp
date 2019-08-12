//============================================================================
// Name    : Dip3.cpp
// Author   : Ronny Haensch
// Version    : 2.0
// Copyright   : -
// Description : 
//============================================================================

#include "Dip3.h"

// Generates gaussian filter kernel of given size
/*
kSize:     kernel size (used to calculate standard deviation)
return:    the generated filter kernel
*/
Mat Dip3::createGaussianKernel(int kSize){

   // TO DO !!!
    Mat GaussianKernel = Mat::zeros(kSize, kSize, CV_32FC1);
    float sum = 0.0;
    float sigma = kSize / 5;
    float mean = kSize / 2;

    for (int row = 0; row < kSize; row++)
    {
        for (int col = 0; col < kSize; col++)
        {
            float mpSigma = sigma * sigma;
            
            float kernelValue = (1/(2 * CV_PI * mpSigma)) * exp(-(pow((row - mean), 2) / mpSigma + pow((col - mean), 2) / (2 * mpSigma)));
            
            GaussianKernel.at<float>(row, col) = kernelValue;
            
            sum += kernelValue;
        }
    }

    for (int row = 0; row < kSize; row++)
    {
        for (int col = 0; col < kSize; col++)
        {
            GaussianKernel.at<float>(row, col) /= sum;
        }
    }
    return GaussianKernel;
}


// Performes a circular shift in (dx,dy) direction
/*
in       input matrix
dx       shift in x-direction
dy       shift in y-direction
return   circular shifted matrix
*/
Mat Dip3::circShift(const Mat& in, int dx, int dy){

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

//Performes convolution by multiplication in frequency domain
/*
in       input image
kernel   filter kernel
return   output image
*/
Mat Dip3::frequencyConvolution(const Mat& in, const Mat& kernel){

   // TO DO !!!
    
    Mat kernelExp = Mat::zeros(in.size(), in.type());
    for (int x = 0; x < kernel.cols; x++)
    {
        for (int y = 0; y < kernel.rows; y++)
        {
            kernelExp.at<float>(y, x) = kernel.at<float>(y, x);
        }
    }
    
    Mat kernelExpStfd = circShift(kernelExp, -(kernel.cols / 2), -(kernel.rows / 2));
    
    Mat dftInputImg = Mat(in.size(), in.type());
    Mat dftkernelExpStfd = Mat(kernelExpStfd.size(), in.type());
    
    dft(in, dftInputImg, 0);
    dft(kernelExpStfd, dftkernelExpStfd);
    
    Mat mulSpcRlt = Mat(in.size(), in.type());
    mulSpectrums(dftInputImg, dftkernelExpStfd, mulSpcRlt, 0);
    
    Mat finalImg = Mat(in.size(), in.type());
    dft(mulSpcRlt, finalImg, DFT_INVERSE + DFT_SCALE);
    
    return finalImg;
}

// Performs UnSharp Masking to enhance fine image structures
/*
in       the input image
type     integer defining how convolution for smoothing operation is done
         0 <==> spatial domain; 1 <==> frequency domain; 2 <==> seperable filter; 3 <==> integral image
size     size of used smoothing kernel
thresh   minimal intensity difference to perform operation
scale    scaling of edge enhancement
return   enhanced image
*/
Mat Dip3::usm(const Mat& in, int type, int size, double thresh, double scale){

   // some temporary images 
   Mat tmp(in.rows, in.cols, CV_32FC1);
   
   // calculate edge enhancement

   // 1: smooth original image
   //    save result in tmp for subsequent usage
   switch(type){
      case 0:
         tmp = mySmooth(in, size, 0);
         break;
      case 1:
         tmp = mySmooth(in, size, 1);
         break;
      case 2: 
	tmp = mySmooth(in, size, 2);
        break;
      case 3: 
	tmp = mySmooth(in, size, 3);
        break;
      default:
         GaussianBlur(in, tmp, Size(floor(size/2)*2+1, floor(size/2)*2+1), size/5., size/5.);
   }

   // TO DO !!!
    
    tmp = tmp - in;
    tmp = tmp * scale;
    
    Mat enhancedImg = in + tmp;
    
    for (int x = 0; x < in.cols; x++)
    {
        for (int y = 0; y < in.rows; y++)
        {
            float pxl;
            
            if (tmp.at<float>(y, x) < thresh)
            {
                pxl = in.at<float>(y, x);
            }
            
            else
            {
                pxl = enhancedImg.at<float>(y, x);
            }
            enhancedImg.at<float>(y, x) = pxl;
        }
    }
    
    return enhancedImg;
    
    
    
   //return in;

}

// convolution in spatial domain
/*
src:    input image
kernel:  filter kernel
return:  convolution result
*/
Mat Dip3::spatialConvolution(const Mat& src, const Mat& kernel){

   // Hopefully already DONE
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
   //return src;

}

// convolution in spatial domain by seperable filters
/*
src:    input image
size     size of filter kernel
return:  convolution result
*/
Mat Dip3::seperableFilter(const Mat& src, int size){

   // optional

   return src;

}

// convolution in spatial domain by integral images
/*
src:    input image
size     size of filter kernel
return:  convolution result
*/
Mat Dip3::satFilter(const Mat& src, int size){

   // optional

   return src;

}

/* *****************************
  GIVEN FUNCTIONS
***************************** */

// function calls processing function
/*
in       input image
type     integer defining how convolution for smoothing operation is done
         0 <==> spatial domain; 1 <==> frequency domain
size     size of used smoothing kernel
thresh   minimal intensity difference to perform operation
scale    scaling of edge enhancement
return   enhanced image
*/
Mat Dip3::run(const Mat& in, int smoothType, int size, double thresh, double scale){

   return usm(in, smoothType, size, thresh, scale);

}


// Performes smoothing operation by convolution
/*
in       input image
size     size of filter kernel
type     how is smoothing performed?
return   smoothed image
*/
Mat Dip3::mySmooth(const Mat& in, int size, int type){

   // create filter kernel
   Mat kernel = createGaussianKernel(size);
 
   // perform convoltion
   switch(type){
     case 0: return spatialConvolution(in, kernel);	// 2D spatial convolution
     case 1: return frequencyConvolution(in, kernel);	// 2D convolution via multiplication in frequency domain
     case 2: return seperableFilter(in, size);	// seperable filter
     case 3: return satFilter(in, size);		// integral image
     default: return frequencyConvolution(in, kernel);
   }
}

// function calls basic testing routines to test individual functions for correctness
void Dip3::test(void){

   test_createGaussianKernel();
   test_circShift();
   test_frequencyConvolution();
   cout << "Press enter to continue"  << endl;
   cin.get();

}

void Dip3::test_createGaussianKernel(void){

   Mat k = createGaussianKernel(11);
   
   if ( abs(sum(k).val[0] - 1) > 0.0001){
      cout << "ERROR: Dip3::createGaussianKernel(): Sum of all kernel elements is not one!" << endl;
      return;
   }
   if (sum(k >= k.at<float>(5,5)).val[0]/255 != 1){
      cout << "ERROR: Dip3::createGaussianKernel(): Seems like kernel is not centered!" << endl;
      return;
   }
   cout << "Message: Dip3::createGaussianKernel() seems to be correct" << endl;
}

void Dip3::test_circShift(void){
   
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
      cout << "ERROR: Dip3::circShift(): Result of circshift seems to be wrong!" << endl;
      return;
   }
   cout << "Message: Dip3::circShift() seems to be correct" << endl;
}

void Dip3::test_frequencyConvolution(void){
   
   Mat input = Mat::ones(9,9, CV_32FC1);
   input.at<float>(4,4) = 255;
   Mat kernel = Mat(3,3, CV_32FC1, 1./9.);

   Mat output = frequencyConvolution(input, kernel);
   
   if ( (sum(output < 0).val[0] > 0) or (sum(output > 255).val[0] > 0) ){
      cout << "ERROR: Dip3::frequencyConvolution(): Convolution result contains too large/small values!" << endl;
      return;
   }
   float ref[9][9] = {{0, 0, 0, 0, 0, 0, 0, 0, 0},
                      {0, 1, 1, 1, 1, 1, 1, 1, 0},
                      {0, 1, 1, 1, 1, 1, 1, 1, 0},
                      {0, 1, 1, (8+255)/9., (8+255)/9., (8+255)/9., 1, 1, 0},
                      {0, 1, 1, (8+255)/9., (8+255)/9., (8+255)/9., 1, 1, 0},
                      {0, 1, 1, (8+255)/9., (8+255)/9., (8+255)/9., 1, 1, 0},
                      {0, 1, 1, 1, 1, 1, 1, 1, 0},
                      {0, 1, 1, 1, 1, 1, 1, 1, 0},
                      {0, 0, 0, 0, 0, 0, 0, 0, 0}};
   for(int y=1; y<8; y++){
      for(int x=1; x<8; x++){
         if (abs(output.at<float>(y,x) - ref[y][x]) > 0.0001){
            cout << "ERROR: Dip3::frequencyConvolution(): Convolution result contains wrong values!" << endl;
            return;
         }
      }
   }
   cout << "Message: Dip3::frequencyConvolution() seems to be correct" << endl;
}
