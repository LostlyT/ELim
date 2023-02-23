#include <stdio.h>
#include <iostream>
#include <stack>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const uchar max_region_num = 80;
const double min_region_area_factor = 0.01;
const Point pointShift2D[8] = 
{
    Point(-1,-1),
    Point(-1, 0),
    Point(-1, 1),
    Point(0, -1),
    Point(0, 1),
    Point(1, -1),
    Point(1, 0),
    Point(1, 1)
};

void growRegion(Mat &src, Mat &dest, Mat&mask, Point seed, int threshold);

int main(int argc, char **argv){

    Mat image = imread(argv[1], IMREAD_GRAYSCALE);

    if(image.empty()) return -1;

    if(image.rows > 650 || image.cols > 650){
        resize(image, image, Size(0,0), 0.5, 0.5);
        //Resize per velocità
    }
    imshow("Immagine originale", image);
    
    int myThreshold=100;

    int min_region_area = (min_region_area_factor * image.cols * image.rows);
    //Size dell'area più piccola che vogliamo considerare

    uchar label = 1; //Label di ogni area della nostra immagine

    //Immagine di destinazione dove ci saranno le varie aree con diversi label
    // 0 sta per indeterminato, 255 sta per ignorato.
    Mat dest = Mat::zeros(image.rows, image.cols, CV_8UC1);

    //Maschera d'aappoggio che iterativamente serve a trovare le regioni
    Mat mask = Mat::zeros(image.rows, image.cols, CV_8UC1);

    //Scorriamo tutta la matrice in cerca di pixel NON semi
    //li rendiamo tali e facciamo accrescere la regione
    for(int x=0; x<image.rows; x++){
        for(int y=0; y<image.cols; y++){

            if(dest.at<uchar>(x, y) == 0){// SE TROVIAMO UN PIXEL IGNORATO, LO RENDIAMO SEME
                growRegion(image, dest, mask, Point(y, x), myThreshold);

                int mask_area = (int)sum(mask).val[0];  //Prendiamo l'area (numero di pixel) della regione
                if(mask_area > min_region_area){        //Se è abbastanza grande
                    dest += mask * label;               //Allora la registriamo
                    cv::imshow("mask", mask*255);
                    cv::waitKey();
                    if(++label > max_region_num){ printf("Numero di regioni massime raggiunto\n"); return -1;}
                }
                else
                {
                    dest += mask * 255;                 //Se è troppo piccola la registriamo come ignorata
                }
                
            }
            
            mask -= mask; //riazzeriamo la maschera

        }
    }

    imshow("fine", dest);
    waitKey(0);
    destroyAllWindows();
    return 0;
}

void growRegion(Mat &src, Mat &dest, Mat&mask, Point seed, int threshold){

    stack<Point> point_stack;
    point_stack.push(seed);

    while(!point_stack.empty()){

        Point center = point_stack.top(); //Considero il pixel top dello stack
        mask.at<uchar>(center) = 1; //Fa parte della regione
        point_stack.pop();

        for(int i=0; i<8; i++){
        //Considero l'8 intorno

            Point curr_point = center+pointShift2D[i];
            if(curr_point.x < 0 || curr_point.y < 0 || curr_point.x > src.rows || curr_point.y > src.cols){//Se sfora i range dell'immagine
                continue;//Passa al prossimo punto
            }
            else{//Altrimenti se è nei giusti range
                uchar delta = (uchar) abs(src.at<uchar>(center) - src.at<uchar>(curr_point));

                if(delta < threshold
                    && dest.at<uchar>(curr_point) == 0
                    && mask.at<uchar>(curr_point) == 0)
                {
                    //Se il predicato  è vero, se il punot non fa già parte di questa o di un'altra regione
                    //allora possiamo proseguire
                    mask.at<uchar>(curr_point) = 1;
                    point_stack.push(curr_point);

                }
            }
        }
        //Fine ciclo 8-intorno


    }

}