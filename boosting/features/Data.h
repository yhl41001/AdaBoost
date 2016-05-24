/*
 * Data.h
 *
 *  Created on: 15/mar/2016
 *      Author: lorenzocioni
 */

#ifndef FEATURE_DATA_H_
#define FEATURE_DATA_H_

#include <vector>
#include <string>
#include <iostream>

using namespace std;

class Data {
private:
	vector<double> features;
	int label;
	int clas;
	double weight;

public:
	Data(vector<double> features);
	Data(vector<double> features, int label);
	Data(vector<double> features, int label, int clas);
	~Data();
	void print();
	const vector<double>& getFeatures() const;
	void setFeatures(const vector<double>& features);
	int getLabel() const;
	void setLabel(int label);
	double getWeight() const;
	void setWeight(double weight);
	int getClas() const;
	void setClas(int clas);
};

#endif /* FEATURES_FEATURE_H_ */
