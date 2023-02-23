#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class Tnode{
    private:
        float mean, stddev;
        Rect rect;
        vector<Tnode*>childs = vector<Tnode*>(4, NULL);
        vector<bool> merged = vector<bool>(4, false);

    public:
        Tnode(Rect rect){ this->rect = rect;}

        float getMean(){return mean;}
        float getStddev(){return stddev;}
        Rect getRect(){return rect;}
        vector<Tnode*>&getChilds(){return childs;}
        vector<bool>&getMerged(){return merged;}
        void setMeanStddev(float mean, float stddev){this->mean=mean; this->stddev=stddev;}

};

Tnode* Split(Mat src, Rect rect, int minDim, float thresh){

    Tnode* currentNode = new Tnode(rect);
    Scalar mean, stddev;
    meanStdDev(src(rect), mean, stddev);

    currentNode->setMeanStddev(mean.val[0], stddev.val[0]);

    if(rect.height > minDim && currentNode->getStddev() >= thresh){
        int dim = rect.width/2;
        vector<Tnode*> &childs = currentNode->getChilds();

        childs[0]=Split(src, Rect(rect.x,rect.y,dim,dim),minDim,thresh);
        childs[1]=Split(src, Rect(rect.x,rect.y+dim,dim,dim),minDim,thresh);
        childs[2]=Split(src, Rect(rect.x+dim,rect.y,dim,dim),minDim,thresh);
        childs[3]=Split(src, Rect(rect.x+dim,rect.y+dim,dim,dim),minDim,thresh);

    }

    return currentNode;

}

void Merge(Tnode* currNode, int minDim, float thresh){

    if(currNode->getRect().height > minDim && currNode->getStddev() >= thresh){

        int k=0;
        vector<Tnode*> &childs = currNode->getChilds();
        vector<bool>& merged = currNode->getMerged();

        for(int i=0; i<4; i++){
            if(childs[i]->getStddev()>= thresh || k==3){
                Merge(childs[i], minDim, thresh);
                merged[i]=false;
            }
            else{
                merged[i]=true;
                k++;
            }
        }
    }
}

void Segmentation(Tnode* currNode, Mat& dest, int minDim, float thresh){

    if(currNode->getRect().height>minDim && currNode->getStddev()>=thresh){//???

        vector<Tnode*>& childs= currNode->getChilds();
        vector<bool>& merged=currNode->getMerged();

        float mean = 0;
        int k = 0;

        for(int i=0; i<4; i++){
            if(merged[i]){
                mean+=childs[i]->getMean();
                k++;
            }
        }

        mean/=k;

        for(int i=0; i<4; i++){
            if(merged[i])
                dest(childs[i]->getRect()) = mean;
            else
                Segmentation(childs[i], dest, minDim, thresh);
        }
    }
    else{
        dest(currNode->getRect()) = currNode->getMean();
    }

}

int main(int argc, char** argv){

    Mat image = imread(argv[1]);

    GaussianBlur(image,  image, Size(5,5), 1, 1);

    int esponente = log2(min(image.rows, image.cols));
    int size = pow(2, esponente);

    Rect rect(0, 0, size, size);
    image = image(rect);//resize dell'immagine ad un quadrato

    int minDim = atoi(argv[2]);
    float thresh = atoi(argv[3]);
    Tnode* root = Split(image, rect, minDim, thresh);

    Merge(root, minDim, thresh);

    Mat dest(image.rows, image.cols, image.type());
    Segmentation(root, dest, minDim, thresh);

    printf("qui\n");

    imshow("",image);
    imshow("", dest);
    waitKey(0);

    return 0;
}
