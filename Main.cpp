/*
 * Main.cpp
 *
 *  Created on: 09/mar/2016
 *      Author: lorenzocioni
 */

#include <iostream>
#include <string>
#include <vector>
#include "AdaBoost.h"
#include "features/Feature.h"

using namespace std;

int main( int argc, char** argv ){

	cout << "AdaBoost classifier" << endl;

	vector<Feature> features = {
		*(new Feature(vector<double>{1, 1}, 1)),
		*(new Feature(vector<double>{1, 2}, 1))
//		*(new Feature(vector<double>{2, 1.5}, 1)),
//		*(new Feature(vector<double>{3, 2}, 1)),
//		*(new Feature(vector<double>{2.8, 4}, -1)),
//		*(new Feature(vector<double>{3.2, 1}, -1)),
//		*(new Feature(vector<double>{3, 3.5}, -1)),
//		*(new Feature(vector<double>{4, 1.5}, 1)),
//		*(new Feature(vector<double>{4.2, 4}, 1))
	};

	AdaBoost* boost = new AdaBoost(features, 20);
	boost->train();

	int p = boost->predict(*(new Feature(vector<double>{5, 4})));

	cout << p << endl;
	delete boost;
    return 0;
}
