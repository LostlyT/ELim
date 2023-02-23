#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

Mat calcNormHist(Mat src){

    Mat normHist = Mat::zeros(1, 256, CV_32FC1);
    //Array visto come matrice di una riga e 256 colonna (che sono i livelli di grigio)

    for(int i=0; i<src.rows; i++){
        for(int j=0; j<src.cols; j++){
            normHist.at<float>(0, src.at<uchar>(i, j))++;
        }
    }

    //Normalizziamo
    for(int i=0; i<normHist.cols; i++)
        normHist.at<float>(0, i) /= src.rows*src.cols;

    return normHist;
}

Mat cumulativeSums(Mat hist){

    Mat p1 = Mat::zeros(1, 256, CV_32FC1);

    for(int k=0; k<256; k++){
        for(int i=0; i<=k; i++){
            p1.at<float>(0, k) += hist.at<float>(0, i); 
            //La probabilita' di un pixel k di appartenere alla Classe 1
            //è uguale alla somma dei pi da 0 a k
        }
    }
    
    return p1;
}

Mat cumulativeMeans(Mat hist){

    Mat m = Mat::zeros(1, 256, CV_32FC1);

    for(int k=0; k<256; k++){
        for(int i=0; i<=k; i++){
            m.at<float>(0, k) += i * hist.at<float>(0, i);
        }
    }

    return m;
}

double globalIntensityMean(Mat hist){

    double tot = 0;

    for(int i=0; i<256; i++)
        tot += i * hist.at<float>(0, i);

    return tot;
}

double globalVariance(Mat hist, double globalMean){

    double sigmaG = 0;

    for(int i=0; i<256; i++){
        sigmaG += pow(i-globalMean, 2) * hist.at<float>(0, i);
    }

    return sigmaG;

}

Mat interclassVariance(Mat p1, double globalMean, Mat cumulativeMeans){

    Mat interVar = Mat::zeros(1, 256, CV_32FC1);
    float p1k, p2k, cumMeanK;
    for(int k=0; k<256; k++){
        p1k = p1.at<float>(0, k);
        p2k = 1 - p1.at<float>(0, k);
        cumMeanK = cumulativeMeans.at<float>(0, k);
        
        if(p1k == 0 || p1k == 1)//Per evitare divisioni per 0
            interVar.at<float>(0, k) = pow(globalMean * p1k - cumMeanK, 2) / 0.0000001;
        else
            interVar.at<float>(0, k) = pow(globalMean * p1k - cumMeanK, 2) / (p1k * p2k);

        printf("interclassVariance(%d) = %f\n", k, interVar.at<float>(0, k));
    }

    return interVar;
}


void OtsuThresholding(Mat src){

    src.convertTo(src, CV_8UC1);

    GaussianBlur(src, src, Size(3, 3), BORDER_DEFAULT);

    Mat laplImage;
    Laplacian(src, laplImage, CV_8UC1, 3, 1, 0, BORDER_DEFAULT);//Si applica il Laplaciano per considerare i pixel vicino a gli edge
    
    src.mul(laplImage);

    threshold(src, src, 200, 255, CV_THRESH_BINARY_INV);

    Mat normHist = calcNormHist(src); //Calcolo istogramma normalizzato

    Mat p1 = cumulativeSums(normHist); //Calcolo le somme cumulative P1 per ogni k e avrò P2 come 1-P1(k)

    Mat cumMeans = cumulativeMeans(normHist); //Calcolo le medie cumulative per ogni k

    double globMean = globalIntensityMean(normHist); //Media d'intensità globale

    Mat interVar = interclassVariance(p1, globMean, cumMeans); // Calcolo la varianza interclasse

    //Ora devo trovare la soglia k*, ovvero il valore k per cui sigma(k) è massimo
    int kstar = 0;
    float maxVal = 0;

    for(int i=0; i<256; i++){
        if(interVar.at<float>(0, i) > maxVal){
            kstar = i;
            maxVal = interVar.at<float>(0, i);
        }
    }

    //Una volta trovato il valore che massimizza la varianza interclasse calcolo
    //il valore di separabilità eta(k)

    int eta_k = (int)interVar.at<float>(0, kstar) / globalVariance(normHist, globMean);

    printf("sigmaG = %f\nkstar = %d\n", interVar.at<float>(0, kstar), kstar);

    threshold(src, src, eta_k, 255, CV_THRESH_BINARY_INV);
}

int main(int argc, char **argv){

    Mat image = imread(argv[1], IMREAD_GRAYSCALE);
    
    if(image.empty()) exit(-1);

    Mat thresh;
    image.copyTo(thresh);

    imshow("Immagine originale", image);
    OtsuThresholding(thresh);

    imshow("Otsu", thresh);
    waitKey(0);
    destroyAllWindows();
    return 0;


}