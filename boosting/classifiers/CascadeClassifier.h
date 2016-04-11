/*
 * CascadeClassifier.hpp
 *
 *  Created on: 11/apr/2016
 *      Author: lorenzocioni
 */

#ifndef BOOSTING_CLASSIFIERS_CASCADECLASSIFIER_H_
#define BOOSTING_CLASSIFIERS_CASCADECLASSIFIER_H_

#include <vector>
#include "Stage.h"

using namespace std;

class CascadeClassifier {
private:
	vector<Stage> stages;

public:
	CascadeClassifier();
	void addStage(Stage stage);
	void train();
	int predict(Data x);
	vector<int> predict(vector<Data> x);
	~CascadeClassifier();
};



#endif /* BOOSTING_CLASSIFIERS_CASCADECLASSIFIER_H_ */
