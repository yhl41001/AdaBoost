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

void CascadeClassifier::addStage(Stage stage){
	this->stages.push_back(stage);
}

int CascadeClassifier::predict(Data x){
	for(int i = 0; i < stages.size(); ++i){
		if(stages[i].predict(x) != 1){
			return false;
		}
	}
	return true;
}

vector<int> CascadeClassifier::predict(vector<Data> x){
	vector<int> output;
	for(int i = 0; i < x.size(); ++i){
		output.push_back(predict(x[i]));
	}
	return output;
}

CascadeClassifier::~CascadeClassifier(){}


