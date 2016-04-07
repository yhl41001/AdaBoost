/*
 * HaarFeature.cpp
 *
 *  Created on: 05/apr/2016
 *      Author: lorenzocioni
 */

#include "HaarFeatures.h"

HaarFeatures::HaarFeatures(){}

vector<double> HaarFeatures::extractFeatures(Mat real, IntegralImage img, int r, int c){
	vector<double> features;
	double white, black;
	/**
	 * Compute feature type (a): horizontal, white left, black right
	 */

	int count = 0;

	for(int w = 1; w < WINDOW/2; ++w){
		for(int i = 0; i < WINDOW - 2 * w + 1; ++i){
			for(int j = 0; j < WINDOW - w + 1; ++j){
				Mat tmp;
				real.copyTo(tmp);

				rectangle(tmp, Rect(j + c, i + r, w, w), Scalar(255, 255, 255), CV_FILLED);
				rectangle(tmp, Rect(j + c + w, i + r, w, w), Scalar(40, 40, 40), CV_FILLED);

				imshow("img", tmp);
				waitKey(10);

//
//								white = img.computeArea(Rect(i + r, j + c, h, w));
//							    black = img.computeArea(Rect(i + r, j + c + w, h, w));
//							    features.push_back((double) (white - black));
								count++;
							    cout << "extracted feature: " << count << endl;

			}
		}
	}


	return features;
}

HaarFeatures::~HaarFeatures(){}

