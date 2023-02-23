#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

void myHoughRette(Mat src, Mat &dst, int votesThreshold){

    double rho;
    double tetha;

    Mat canny;
    //Effettuiamo un  po' di smoothing sull'immagine per attenuare il rumore
    GaussianBlur(src, src, Size(5,5),  0, 0, BORDER_DEFAULT);

    //Innanzitutto applichiamo Canny per avere l'immagine della segmentazione e quindi tutti gli edge
    Canny(src, canny, 80, 110, 3);
    canny.copyTo(dst);
    int dist = hypot(src.rows, src.cols); //hypot() Calcola la massima distanzza possibile tra due punti dato un numero di righe e colonne
                                          //Sarebbe sostanzialmente l'ipotenusa

    //Inizializziamo la matrice dei "voti"
    //Sulle righe avremo rho, e sulle colonne tetha
    Mat votes = Mat::zeros(dist*2, 180, CV_8U);
    
    //Scansioniamo ogni singolo pixel di edge
    for(int x=0; x<src.rows; x++){
        for(int y=0; y<src.cols; y++){
            if(canny.at<uchar>(x, y) == 255){//ANDIAMO A CONSIDERARE SOLO I PUNTI DI EDGE
                for(tetha=0; tetha<180; tetha++){//VOTIAMO PER LA SINUSOIDE DI QUEL PUNTO DI EDGE
                    rho = y*cos((tetha-90)*CV_PI/180) + x*sin((tetha-90)*CV_PI/180);
                    votes.at<uchar>(rho, tetha)++;
                }
            }
        }
    }
    //Finita la fase di voto

    
    //Analizziamo la fase di voto, e controlliamo quali celle superano il th
    for(int r=0; r<votes.rows; r++){
        for(int t=0; t<votes.cols; t++){
            if(votes.at<uchar>(r, t) >= votesThreshold){ // Se i voti sono considerabili, mi calcolo l'angolo tetha e le coordinate x,y
                tetha = (t-90)*CV_PI/180; // Converto da gradi a radianti

                //Calcolo i punti che sono gli estremi della retta
                double sin_t = sin(tetha);
                double cos_t = cos(tetha);

                int x = r * cos_t;
                int y = r * sin_t;

                Point pt1( cvRound(x + dist*(-sin_t)), cvRound(y + dist*cos_t) );
                Point pt2( cvRound(x - dist*(-sin_t)), cvRound(y - dist*cos_t) );

                line(dst, pt1, pt2, Scalar(255), 2, 0);

            }
        }
    }
    printf("arrivo qua\n");
}

int main(int argc, char **argv){
    Mat image = imread(argv[1], IMREAD_GRAYSCALE);

    if(image.empty()) exit(-1);

    imshow("Immagine originale", image);

    Mat rette(image.rows, image.cols, CV_8U);

    myHoughRette(image, rette, 120);

    imshow("Rette", rette);
    waitKey(0);
    destroyAllWindows();

    return 0;

}