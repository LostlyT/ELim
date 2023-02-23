#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>
#include <unistd.h>
#include <math.h>

using namespace std;
using namespace cv;

void hysteresisThresholding(Mat nms, Mat &dst, int min, int max){

    int curr_val;

    for(int i=1; i<nms.rows-1; i++){
        for(int j=1; j<nms.cols-1; j++){
            curr_val = nms.at<uchar>(i,j);
            
            //Determino i pixel forti e annullo quelli deboli
            // pixel forte > max
            // pixel debole < min
            if(curr_val > max)
                dst.at<uchar>(i, j) = 255;
            else if(curr_val < min)
                dst.at<uchar>(i, j) = 0;
            //------
            //Altrimenti se e' un pixel incerto, controllo:
            //Se e' accanto a dei pixel forti, di edge, allora proseguo con l'assegnazione
            //altrimenti e' rumore isolato
            else if( curr_val <= max && curr_val >= min){
                for(int u=-1; u<=1; u++){
                    for(int v=-1; v<=1; v++){
                        if(nms.at<uchar>(i+u, j+v)>max)
                            dst.at<uchar>(i, j)=255;
                    }
                }
            }
        }
    }

}

void nonMaximaSuppression(Mat mag, Mat alpha, Mat &dst){

    float angolo;
    int mag_curr;
    for(int i=1; i<mag.rows-1; i++){
        for(int j=1; j<mag.cols-1; j++){
            mag_curr = mag.at<uchar>(i,j);

            angolo = alpha.at<float>(i,j);

            //normalizziamo l'angolo perche' abbiamo usato phase invece di atan()
            //se e' maggiore di 180 sottraiamo 360 per avere valori minori di zero
            angolo = angolo>180? angolo-360 : angolo;
            //Ora abbiamo le quattro direzioni e controlliamo ogni volta i valori
            //P.S. non effettuiamo "annullamenti", ovvero assegnazioni a zero
            //perche' diamo per scontato che l'immagine di output sia una matrice di zeri
            if( (angolo>-22.5 && angolo<22.5) || (angolo<-157.5 && angolo>157.5)){
                if(mag.at<uchar>(i,j-1) < mag_curr && mag.at<uchar>(i,j+1) < mag_curr)
                    dst.at<uchar>(i,j) = mag_curr;
            }
            else if( (angolo<157.5 && angolo>112.5) || (angolo>-67.5 && angolo<-22.5)){
                if(mag.at<uchar>(i-1, j-1) < mag_curr && mag.at<uchar>(i+1, j+1) < mag_curr)
                    dst.at<uchar>(i,j) = mag_curr;
            }
            else if( (angolo<112.5 && angolo>67.5) || (angolo>-112.5 && angolo<-67.5)){
                if(mag.at<uchar>(i-1, j) < mag_curr && mag.at<uchar>(i+1, j) < mag_curr)
                    dst.at<uchar>(i,j) = mag_curr;
            }
            else if( (angolo<67.5 && angolo>22.5) || (angolo>-157.5 && angolo<-112.5)){
                if( mag.at<uchar>(i+1, j-1) < mag_curr && mag.at<uchar>(i-1,j+1) < mag_curr)
                    dst.at<uchar>(i,j) = mag_curr;
            }
        }
    }
}

void cannySegmentation(Mat src, Mat &dst){

    //Smoothing delll'immagine per eliminare il rumore
    GaussianBlur(src, src, Size(5,5), 0, 0);

    //Calcolo gx
    Mat gx, gx_2;
    Sobel(src, gx, CV_32FC1, 1, 0, 3);

    //Calcolo gy
    Mat gy, gy_2;
    Sobel(src, gy, CV_32FC1, 0, 1, 3);

    //Calcolo la magnitudine
    Mat magnitudine;
    pow(gx, 2, gx_2);
    pow(gy, 2, gy_2);
    sqrt(gx_2+gy_2, magnitudine);
    normalize(magnitudine, magnitudine, 0, 255, NORM_MINMAX, CV_8U); // NORMALIZZO

    //Calcolo le direzioni del gradiente
    Mat alpha;
    phase(gx, gy, alpha, true); // E' uguale ad atan(arg), ma quest'ultima ritorna sia positivi che negativi, phase solo positivi (0 - 359)

    printf("Effettuo la nms\n");
    //Effettuiamo la non maxima suppression
    Mat nms=Mat::zeros(magnitudine.rows, magnitudine.cols, CV_8U);
    nonMaximaSuppression(magnitudine, alpha, nms);

    printf("Effettuiamo la th\n");
    //Effettuiamo il thresholding con isteresi
    hysteresisThresholding(nms, dst, 50, 70);
}

int main(int argc, char**argv){

    if( argc < 2){
        cout<<"usage: "<<argv[0]<<" image_name"<<endl;
        exit(0);
    }

    String imageName = argv[1];

    //Leggo l'immagine
    Mat image;
    image = imread(imageName, IMREAD_GRAYSCALE);
    if( image.empty() ){
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    Mat canny = Mat::zeros(image.rows, image.cols, CV_8U);
    
    cannySegmentation(image,canny);
    printf("fine\n");

    imshow("Immagine originale", image);
    imshow("Canny segmentation", canny);
    waitKey(0);
    destroyAllWindows();

    return 0;
}