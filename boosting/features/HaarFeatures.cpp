/*
 * HaarFeature.cpp
 *
 *  Created on: 05/apr/2016
 *      Author: lorenzocioni
 */

#include "HaarFeatures.h"

void HaarFeatures::getFeature(int size, WeakClassifier* wc){
	Mat img(Size(size, size), CV_8UC3);
	extractFeatures(img, size, 0, 0, true, wc);
}

vector<double> HaarFeatures::extractFeatures(Mat img, int size, int r, int c){
	return extractFeatures(img, size, r, c, false, NULL);
}

/**
 * Extracting haar like feature from an image
 * integralImage: the integral image
 * size: the detection window size
 * r: row offset
 * c: column offset
 */
vector<double> HaarFeatures::extractFeatures(Mat integralImage, int size, int r, int c, bool store, WeakClassifier* wc) {

	int minArea = 36; //minimum area of the haar feature (as in opencv implementation)
	vector<double> features(TOT_FEATURES);
	int count = 0;
	double white, black;
	int x, y, sizeX, sizeY, width, height;

	/**
	 * Extracting feature type (2 x 1)
	 */
	sizeX = 2;
	sizeY = 1;
	for (width = sizeX; width <= size; width += sizeX) {
		for (height = sizeY; height <= size; height += sizeY) {
			for (x = 0; x <= size - width; ++x) {
				for (y = 0; y <= size - height; ++y) {
					if(width * height > minArea){
						white = IntegralImage::computeArea(integralImage, Rect(x + c, y + r, width / 2, height));
						black = IntegralImage::computeArea(integralImage, Rect(x + c + width / 2, y + r, width / 2, height));
						features[count] = (double) (white - black);
						if(store && count == wc->getDimension()){
							wc->setBlacks(vector<Rect>{Rect(x + c + width / 2, y + r, width / 2, height)});
							wc->setWhites(vector<Rect>{Rect(x + c, y + r, width / 2, height)});
							return features;
						}
						count++;
					}
				}
			}
		}
	}
	/**
	 * Extracting feature type (1 x 2)
	 */
	sizeX = 1;
	sizeY = 2;
	for (width = sizeX; width <= size; width += sizeX) {
		for (height = sizeY; height <= size; height += sizeY) {
			for (x = 0; x <= size - width; ++x) {
				for (y = 0; y <= size - height; ++y) {
					if(width * height > minArea){
						white = IntegralImage::computeArea(integralImage, Rect(x + c, y + r, width, height / 2));
						black = IntegralImage::computeArea(integralImage, Rect(x + c, y + r + height / 2, width, height / 2));
						features[count] = (double) (white - black);
						if(store && count == wc->getDimension()){
							wc->setBlacks(vector<Rect>{Rect(x + c, y + r + height / 2, width, height / 2)});
							wc->setWhites(vector<Rect>{Rect(x + c, y + r, width, height / 2)});
							return features;
						}
						count++;
					}
				}
			}
		}
	}
	/**
	 * Extracting feature type (3 x 1)
	 */
	sizeX = 3;
	sizeY = 1;
	for (width = sizeX; width <= size; width += sizeX) {
		for (height = sizeY; height <= size; height += sizeY) {
			for (x = 0; x <= size - width; ++x) {
				for (y = 0; y <= size - height; ++y) {
					if(width * height > minArea){
						white = IntegralImage::computeArea(integralImage, Rect(x + c, y + r, width / 3, height));
						black = IntegralImage::computeArea(integralImage, Rect(x + c + width / 3, y + r, width / 3, height));
						white += IntegralImage::computeArea(integralImage, Rect(x + c + 2 * width / 3, y + r, width / 3, height));
						features[count] = (double) (white - black);
						if(store && count == wc->getDimension()){
							wc->setBlacks(vector<Rect>{Rect(x + c + width / 3, y + r, width / 3, height)});
							wc->setWhites(vector<Rect>{Rect(x + c, y + r, width / 3, height) ,Rect(x + c + 2 * width / 3, y + r, width / 3, height)});
							return features;
						}
						count++;
					}
				}
			}
		}
	}
	/**
	 * Extracting feature type (2 x 2)
	 */
	sizeX = 2;
	sizeY = 2;
	for (width = sizeX; width <= size; width += sizeX) {
		for (height = sizeY; height <= size; height += sizeY) {
			for (x = 0; x <= size - width; ++x) {
				for (y = 0; y <= size - height; ++y) {
					if(width * height > minArea){
						white = IntegralImage::computeArea(integralImage, Rect(x + c, y + r, width / 2, height / 2));
						black = IntegralImage::computeArea(integralImage, Rect(x + c + width / 2, y + r, width / 2, height / 2));
						white += IntegralImage::computeArea(integralImage, Rect(x + c + width / 2, y + r + height / 2, width / 2, height / 2));
						black += IntegralImage::computeArea(integralImage, Rect(x + c, y + r + height / 2, width / 2, height / 2));
						features[count] = (double) (white - black);
						if(store && count == wc->getDimension()){
							wc->setBlacks(vector<Rect>{Rect(x + c + width / 2, y + r, width / 2, height / 2), Rect(x + c, y + r + height / 2, width / 2, height / 2)});
							wc->setWhites(vector<Rect>{Rect(x + c, y + r, width / 2, height / 2), Rect(x + c + width / 2, y + r + height / 2, width / 2, height / 2)});
							return features;
						}
						count++;
					}
				}
			}
		}
	}

	return features;
}

double HaarFeatures::evaluate(Mat intImg, vector<Rect> whites, vector<Rect> blacks){
	double white = 0;
	double black = 0;
	for(int w = 0; w < whites.size(); ++w){
		white += IntegralImage::computeArea(intImg, whites[w]);
	}
	for(int b = 0; b < blacks.size(); ++b){
		black += IntegralImage::computeArea(intImg, blacks[b]);
	}
	return white - black;
}

