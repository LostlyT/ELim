#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

vector<float>& normalizedHistogram(Mat src){

    vector<float>&histogram = *(new vector<float>(256,0));

    for(int i=0; i<src.rows; i++){
        for(int j=0; j<src.cols; j++){
            histogram[src.at<uchar>(i,j)]++;
        }
    }

    for(int i=0; i<256; i++)
        histogram[i] /= (src.rows * src.cols);

        return histogram;
}

int npercentile(Mat src, float n){
    vector<float>&histogram = normalizedHistogram(src);

    float sum=0;
    for(int i=0; i<256; i++){
        sum+=histogram[i];

        if(sum>=n)
            return i;
    }
    return -1;
}

vector<float>& maskNormHist(Mat src, Mat mask){

    vector<float>& hist = *(new vector<float>(256, 0));

    int n = 0;

    for(int i=0; i<src.rows; i++){
        for(int j=0; j<src.cols; j++){

            if(mask.at<uchar>(i,j) == 1){
                hist[src.at<uchar>(i,j)]++;
                n++;
            }
        }
    }

    for(int i=0; i<256; i++)
        hist[i] /= n;

    return hist;
}

int otsu(vector<float> histogram){

    int thresh;

    float mediaGlobale = 0;
    for(int i=0; i<256; i++)
        mediaGlobale += histogram[i] * i;

    float p1 = 0, varianzaIntraclasse, mediaCumulativaK = 0, max = 0;

    for(int k=0; k<256; k++){
        p1 += histogram[k];
        mediaCumulativaK += histogram[k] * k;

        varianzaIntraclasse = pow(mediaGlobale * p1 - mediaCumulativaK, 2)/(p1* (1-p1));

        if(varianzaIntraclasse > max){
            max=varianzaIntraclasse;
            thresh = k;
        }

    }

    return thresh;
}

void otsu2soglie(vector<float> histogram, int k[]){

    float mediaGlobale = 0;
    for(int i=0; i<256; i++)
        mediaGlobale += histogram[i] * i;

    float varianzaIntraclasse, max = 0;
    float p[3] = {0, 0, 0}, medCum[3] = {0, 0, 0};

    for(int i=0; i<256-2; i++){
        p[0] += histogram[i];
        medCum[0] += histogram[i] * i;

        for(int j=i+1; j<256-1; j++){
            p[1] += histogram[j];
            medCum[1] += histogram[j] * j;

            for(int z=j+1; z<256; z++){
                p[2] += histogram[z];
                medCum[2] += histogram[z] * z;

                varianzaIntraclasse = p[0]*pow(medCum[0]-mediaGlobale,2) +
                                        p[1]*pow(medCum[1]- mediaGlobale, 2) +
                                        p[2]*pow(medCum[2] - mediaGlobale, 2);

                if(varianzaIntraclasse > max){
                    max = varianzaIntraclasse;
                    k[0]=i;
                    k[1]=j;
                }
            }

            p[2]=0;
            medCum[2]=0;
        }
        p[1]=0;
        medCum[1]=0;
    }
}

Mat multipleThresh(Mat src, int k[]){
    Mat dest = Mat::zeros(src.rows, src.cols, CV_8UC1);

    for(int i=0; i<src.rows; i++){
        for(int j=0; j<src.cols; j++){
            if(src.at<uchar>(i,j)>k[1])
                dest.at<uchar>(i,j)=255;
            if(src.at<uchar>(i,j) > k[0])
                dest.at<uchar>(i,j)=125;
        }
    }

    return dest;
}

int main(int argc, char **argv){

    Mat src = imread(argv[1], IMREAD_GRAYSCALE);

    GaussianBlur(src, src, Size(7, 7), 1, 1);

    Mat laplacian;
    Laplacian(src, laplacian, CV_32FC1, 3);
    normalize(laplacian, laplacian, 0, 255, NORM_MINMAX, CV_8UC1);

    int thresh = npercentile(laplacian, 0.9);

    Mat mask;
    threshold(laplacian, mask, thresh, 1, CV_8UC1);

    imshow("src", src);
    imshow("masked", src.mul(mask));

    vector<float>&histogram = maskNormHist(src, mask);
    waitKey(0);

    //HO L'ISTOGRAMMA NORMALIZZATO SOLO SUI PIXEL DI EDGE E POSSO PROSEGUIRE CON OTSU

    thresh = otsu(histogram);
    Mat otsu1Soglia;
    threshold(src, otsu1Soglia, thresh, 255, THRESH_BINARY_INV);
    imshow("otsu co na soglia", otsu1Soglia);
    waitKey(0);

    int k[2] = {0, 0};
    otsu2soglie(histogram, k);
    Mat dest = multipleThresh(src, k);

    imshow("2soglie", dest);
    waitKey(0);

    return 0;

}