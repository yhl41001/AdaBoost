/*
 * Data.cpp
 *
 *  Created on: 15/mar/2016
 *      Author: lorenzocioni
 */

#include "Data.h"

Data::Data(vector<double> features, int label):
	features(features),
	label(label),
	weight(0.),
	clas(1){}

Data::Data(vector<double> features, int label, int clas):
	features(features),
	label(label),
	weight(0.),
	clas(clas){}

Data::Data(vector<double> features):
	features(features),
	label(0),
	weight(0.),
	clas(1){}

Data::~Data() {
	features.clear();
}

/**
 * Print the vector of features
 */
void Data::print(){
	cout << "[";
	for(int i = 0; i < features.size(); ++i){
		cout << features[i];
		if(i < features.size() - 1){
			cout << ", ";
		}
	}
	cout << "] (label: " << label << ", weight: " << weight << ")" << endl;
}

const vector<double>& Data::getFeatures() const {
	return features;
}

void Data::setFeatures(const vector<double>& features) {
	this->features = features;
}

int Data::getLabel() const {
	return label;
}

double Data::getWeight() const {
	return weight;
}

void Data::setWeight(double weight) {
	this->weight = weight;
}

void Data::setLabel(int label) {
	this->label = label;
}

int Data::getClas() const {
	return clas;
}

void Data::setClas(int clas) {
	this->clas = clas;
}
