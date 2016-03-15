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
	Feature* f = new Feature(vector<double>(3, 1));

	features.push_back(*f);

	AdaBoost* boost = new AdaBoost(features, vector<int>(1, 1), 20);
	boost->train();


	delete boost;
    return 0;
}
