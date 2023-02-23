#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void initializeClusters(Mat src, int clusters_number, vector<Scalar> &clusters_centers, vector< vector<Point>>& ptInClusters){

	RNG random(cv::getTickCount());

	for(int i=0; i<clusters_number; i++){

		Point center;
		center.x = random.uniform(0, src.cols);
		center.y = random.uniform(0, src.rows);

		//Prendiamo i valori di R, G e B del pixel casuale, e lo salviamo come centro con 3 diverse feature
		Scalar centerPixel = src.at<Vec3b>(center.y, center.x);
		Scalar centerk(centerPixel.val[0], centerPixel.val[1], centerPixel.val[2]);
		clusters_centers.push_back(centerk);

		//inizializziamo anche il vettore associato ai punti nel cluster
		vector<Point> ptInClusterK;
		ptInClusters.push_back(ptInClusterK);
	}
}

double computeDistance(Scalar pixel, Scalar center){
	double distBlu = pixel.val[0] - center.val[0];
	double distGreen = pixel.val[1] - center.val[1];
	double distRed = pixel.val[2] - center.val[2];

	return sqrt(pow(distBlu, 2)+pow(distGreen, 2)+pow(distRed, 2));
	
}


void associatePixelsToClusters(Mat src, int clusters_number, vector<Scalar> &clusters_centers, vector< vector<Point>>& ptInClusters){

	//Per ogni pixel trovo il cluster associato
	for(int x=0; x<src.rows; x++){
		for(int y=0; y<src.cols; y++){

			double minDistance = INFINITY;
			int closestCluster = 0;

			Scalar pixel = src.at<Vec3b>(x, y);

			for(int k=0; k<clusters_number; k++){

				double distance = computeDistance(pixel, clusters_centers[k]);

				if(distance < minDistance){
					minDistance = distance;
					closestCluster = k;
				}
			}

			//Una volta uscito ho trovato il cluster col centro più vicino e lo assegno
			ptInClusters[closestCluster].push_back(Point(y,x));

		}		
	}

}

double adjustClusterCenters(Mat src, int clusters_number, vector<Scalar> &clusters_centers, vector< vector<Point>>& ptInClusters, double oldCenter, double newCenter){

	double diffChange;

	for(int k=0; k<clusters_number; k++){

		double newBlue = 0;
		double newGreen = 0;
		double newRed = 0;

		for(int i=0; i<ptInClusters[k].size(); i++){
			
			newBlue += src.at<Vec3b>(ptInClusters[k][i])[0];
			newGreen += src.at<Vec3b>(ptInClusters[k][i])[1];
			newRed += src.at<Vec3b>(ptInClusters[k][i])[2];
		}

		newBlue /= ptInClusters[k].size();
		newGreen /= ptInClusters[k].size();
		newRed /= ptInClusters[k].size();

		Scalar newPixel(newBlue, newGreen, newRed);

		//Calcoliamo lo spostamento medio dei nuovi centri
		newCenter += computeDistance(newPixel, clusters_centers[k]);

		clusters_centers[k] = newPixel;

	}


	newCenter /= clusters_number;

	diffChange = abs(oldCenter-newCenter);
	printf("diffChange is: %f\n", diffChange);

	return diffChange;

}

Mat applyFinalCluster(Mat src, int clusters_number, vector<Scalar> &clusters_centers, vector< vector<Point>>& ptInClusters){

	Mat result = src.clone();;

	for(int k=0; k<clusters_number; k++){

		for(int i=0; i<ptInClusters[k].size(); i++){

			src.at<Vec3b>(ptInClusters[k][i])[0] = clusters_centers[k].val[0];
			src.at<Vec3b>(ptInClusters[k][i])[1] = clusters_centers[k].val[1];
			src.at<Vec3b>(ptInClusters[k][i])[2] = clusters_centers[k].val[2];
		}

	}

	return result;

}

int main(int argc, char **argv){
	
	Mat image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	if(image.empty()) return -1;

	int clusters_number = atoi(argv[2]);

	//Inizializziamo le variabili utili
	vector<Scalar> clusters_centers;
	vector< vector<Point>> ptInClusters;

	double threshold = 0.1;
	double oldCenter = INFINITY;
	double newCenter = 0;
	double diffChange = oldCenter - newCenter;

	//Inizializzo i cluster
	initializeClusters(image, clusters_number, clusters_centers, ptInClusters);

	while(diffChange > threshold){

		newCenter = 0;

		//resetto i pixel nei cluster che ritroveranno una nuova posizione
		for(int k=0; k<clusters_number; k++) ptInClusters[k].clear();

		associatePixelsToClusters(image, clusters_number, clusters_centers, ptInClusters);

		diffChange = adjustClusterCenters(image, clusters_number, clusters_centers, ptInClusters, oldCenter, newCenter);
		printf("iterazione\n");
	}

	Mat km = image.clone();

	km = applyFinalCluster(image, clusters_number, clusters_centers, ptInClusters);
	imshow("Risultatoo finale", km);
	imshow("immagine originale", image);
	waitKey(0);
	destroyAllWindows();

	return 0;
}


// SOTTO CE LA MIA VERSIONEEEEEEEEEEEE
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class Cluster{
		
	public:
		vector<Point> points;
		int b, g, r;
		Cluster(Mat src){
			RNG random(cv::getTickCount());

			int x = random.uniform(0, src.rows);
			int y = random.uniform(0, src.cols);

			b = src.at<Vec3b>(x, y)[0];
			g = src.at<Vec3b>(x, y)[1];
			r = src.at<Vec3b>(x, y)[2];
		}
};


void associatePoints(Mat src, vector<Cluster>& clusters){

	double minDist = INFINITY;
	int minClust;

	for(int i=0; i<src.rows; i++){
		for(int j=0; j<src.cols; j++){
			minDist = 100000;
			minClust = 0;
			for(int k=0; k<clusters.size(); k++){

				double delta = sqrt(pow(src.at<Vec3b>(i,j)[0] - clusters[k].b, 2) +
									pow(src.at<Vec3b>(i,j)[1] - clusters[k].g, 2) +
									pow(src.at<Vec3b>(i,j)[2] - clusters[k].r, 2));

				if(delta < minDist){
					minDist = delta;
					minClust = k;
				}

			}

			clusters[minClust].points.push_back(Point(j, i));
		}
	}

}			

double adjustCenters(Mat src, vector<Cluster>& clusters, double& newCenter){

	double diff;
	double oldCenter = newCenter;

	for(int k=0; k<clusters.size(); k++){
		
		double newB = 0, newG = 0, newR = 0;
		printf("size punti del cluster %d = %ld\n", k, clusters[k].points.size());
		for(int i=0; i<clusters[k].points.size(); i++){
			newB += src.at<Vec3b>(clusters[k].points[i])[0];
			newG += src.at<Vec3b>(clusters[k].points[i])[1];
			newR += src.at<Vec3b>(clusters[k].points[i])[2];
		}

		newB /= clusters[k].points.size();
		newG /= clusters[k].points.size();
		newR /= clusters[k].points.size();

		newCenter += sqrt(pow(clusters[k].b - newB, 2) +
				pow(clusters[k].g - newG, 2) +
				pow(clusters[k].r - newR, 2));

		clusters[k].b = newB;
		clusters[k].g = newG;
		clusters[k].r = newR;
	}

	newCenter/=clusters.size();

	diff=abs(oldCenter-newCenter);

	return diff;
}
	
Mat finalizeClusters(Mat src, vector<Cluster> clusters){

	Mat dest(src.rows, src.cols, src.type());

	for(int k=0; k<clusters.size(); k++){

		for(int i=0; i<clusters[k].points.size(); i++){
			dest.at<Vec3b>(clusters[k].points[i])[0] = clusters[k].b;
			dest.at<Vec3b>(clusters[k].points[i])[1] = clusters[k].g;
			dest.at<Vec3b>(clusters[k].points[i])[2] = clusters[k].r;
		}
	}

	return dest;
}

int main(int argc, char **argv){

	Mat src = imread(argv[1]);
	
	int ncluster = atoi(argv[2]);

	vector<Cluster> clusters;

	for(int i=0; i<ncluster; i++)
		clusters.push_back(Cluster(src));
	
	float thresh = 0.1;

	double newCenter = 0;
	double diffChange = 1000000;

	while(diffChange > thresh){

		for(int k=0; k<ncluster; k++) clusters[k].points.clear();

		associatePoints(src, clusters);

		diffChange = adjustCenters(src, clusters, newCenter);
		printf("diffChange = %f\n", diffChange);
	}	

	Mat dest = finalizeClusters(src, clusters);

	imshow("dest", dest);
	waitKey(0);

	return 0;
}

