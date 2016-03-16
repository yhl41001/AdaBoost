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

	vector<Feature> features;

	Feature* f2 = new Feature(vector<double>(3, 2));
	Feature* f3 = new Feature(vector<double>(3, 4));
	Feature* f = new Feature(vector<double>(3, 1));

	features.push_back(*f2);
	features.push_back(*f3);
	features.push_back(*f);

	AdaBoost* boost = new AdaBoost(features, vector<int>(3, 1), 20);
	boost->train();


	delete boost;
    return 0;
}
