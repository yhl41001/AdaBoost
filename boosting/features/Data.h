/*
 * Data.h
 *
 *  Created on: 15/mar/2016
 *      Author: lorenzocioni
 */

#ifndef FEATURE_DATA_H_
#define FEATURE_DATA_H_

#include <vector>
#include <iostream>

using namespace std;

class Data {
private:
	vector<double> features;
	int label;
	double weight;

public:
	Data(vector<double> features);
	Data(vector<double> features, int label);
	~Data();
	void print();
	const vector<double>& getFeatures() const;
	void setFeatures(const vector<double>& features);
	int getLabel() const;
	void setLabel(int label);
	double getWeight() const;
	void setWeight(double weight);
};

#endif /* FEATURES_FEATURE_H_ */
