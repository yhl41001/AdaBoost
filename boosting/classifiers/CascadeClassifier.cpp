/*
 * CascadeClassifier.cpp
 *
 *  Created on: 11/apr/2016
 *      Author: lorenzocioni
 */

#include "CascadeClassifier.h"

CascadeClassifier::CascadeClassifier(){
	this->stages = {};
}

void CascadeClassifier::addStage(Stage* stage){
	stages.push_back(stage);
}

int CascadeClassifier::predict(vector<double> x){
	for(int i = 0; i < stages.size(); ++i){
		if(stages[i]->predict(x) != 1){
			return -1;
		}
	}
	return 1;
}

int CascadeClassifier::predict(vector<Haar> values){
	for(int i = 0; i < stages.size(); ++i){
		if (stages[i]->predict(values) != 1) {
			return -1;
		}
	}
	return 1;
}

CascadeClassifier::~CascadeClassifier() {
}

const vector<Stage*>& CascadeClassifier::getStages() const {
	return stages;
}

void CascadeClassifier::setStages(const vector<Stage*>& stages) {
	this->stages = stages;
}
