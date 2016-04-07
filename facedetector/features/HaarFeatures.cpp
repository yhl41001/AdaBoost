/*
 * HaarFeature.cpp
 *
 *  Created on: 05/apr/2016
 *      Author: lorenzocioni
 */

#include "HaarFeatures.h"

HaarFeatures::HaarFeatures() {
}

vector<double> HaarFeatures::extractFeatures(IntegralImage img, int r, int c) {
	vector<double> features;
	const int types[][2] = { { 2, 1 }, { 1, 2 }, { 3, 1 }, { 1, 3 }, { 2, 2 } };

	int count = 0;
	double white, black;
	int i, x, y, sizeX, sizeY, width, height;
	for (i = 0; i < sizeof(types); i++) {
		sizeX = types[i][0];
		sizeY = types[i][1];
		/* each size (multiples of basic shapes) */
		for (width = sizeX; width <= WINDOW; width += sizeX) {
			for (height = sizeY; height <= WINDOW; height += sizeY) {
				/* each possible position given size */
				for (x = 0; x <= WINDOW - width; x++) {
					for (y = 0; y <= WINDOW - height; y++) {
						white = img.computeArea(Rect(y + r, x + c, width/2, height));
						black = img.computeArea(Rect(y + r, x + c + width/2, width/2, height));
						features.push_back((double) (white - black));
						count++;
					}
				}
			}
		}
	}

	cout << count << endl;

	return features;
}

HaarFeatures::~HaarFeatures() {
}

