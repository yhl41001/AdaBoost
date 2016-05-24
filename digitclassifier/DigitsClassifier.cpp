/*
 * DigitsClassifier.cpp
 *
 *  Created on: 20/mag/2016
 *      Author: lorenzocioni
 */

#include "DigitsClassifier.h"

DigitsClassifier::DigitsClassifier(string imagesPath, string labelsPath, int numImages){
	readMnist(imagesPath, this->digits, numImages);
	readMnistLabels(labelsPath, this->labels, numImages);
	vector<float> test;
	int winSize = digits[0].rows;
	int blockSize = winSize/2;
	int blockStride = blockSize/2;
	int cellSize = blockStride;
	int bins = 22;
	this->hog = new HOGDescriptor(cvSize(winSize, winSize), cvSize(blockSize, blockSize),
					cvSize(blockStride, blockStride), cvSize(cellSize, cellSize), bins);
}

vector<double> DigitsClassifier::extractHOGfeatures(Mat digit){
	vector<float> output;
	hog->compute(digit, output, Size(1,1), Size(0,0));
	vector<double> features(output.begin(), output.end());
	return features;
}

void DigitsClassifier::train(){
	//Extracting HOG features from digits
	vector<Data*> data(digits.size());
	for(unsigned int i = 0; i < digits.size(); ++i){
		data[i] = new Data(extractHOGfeatures(digits[i]), labels[i]);
	}
	boost = new AdaBoostMH(data, 20, 10);
	boost->train();
}

int DigitsClassifier::reverseInt(int i){
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return ((int) ch1 << 24) + ((int) ch2 << 16) + ((int) ch3 << 8) + ch4;
}

void DigitsClassifier::readMnist(string filename, vector<Mat> &images, int numImages){
	cout << "Loaded MNIST digits dataset" << endl;
    ifstream file (filename, ios::binary);
    if (file.is_open()){
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        file.read((char*) &number_of_images, sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);
        if(number_of_images > numImages)
        	number_of_images = numImages;
        file.read((char*) &n_rows, sizeof(n_rows));
        n_rows = reverseInt(n_rows);
        file.read((char*) &n_cols, sizeof(n_cols));
        n_cols = reverseInt(n_cols);
        for(int i = 0; i < number_of_images; ++i){
            cv::Mat tp = Mat::zeros(n_rows, n_cols, CV_8UC1);
            for(int r = 0; r < n_rows; ++r){
                for(int c = 0; c < n_cols; ++c){
                    unsigned char temp = 0;
                    file.read((char*) &temp, sizeof(temp));
                    tp.at<uchar>(r, c) = (int) temp;
                }
            }
            images.push_back(tp);
        }
    }
}

void DigitsClassifier::readMnistLabels(string filename, vector<double> &vector, int numImages){
	cout << "Loaded MNIST digits labels" << endl;
    ifstream file (filename, ios::binary);
    if (file.is_open()){
        int magic_number = 0;
        int number_of_images = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);
        if(number_of_images > numImages)
        	number_of_images = numImages;
        for(int i = 0; i < number_of_images; ++i){
            unsigned char temp = 0;
            file.read((char*) &temp, sizeof(temp));
            vector.push_back((double) temp);
        }
    }
}

DigitsClassifier::~DigitsClassifier(){}
