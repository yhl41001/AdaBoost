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
	vector<Stage*> stages;

public:
	CascadeClassifier();
	void addStage(Stage* stage);
	void train();
	int predict(Mat img);
	int predict(vector<double> x);
	~CascadeClassifier();
	const vector<Stage*>& getStages() const;
	void setStages(const vector<Stage*>& stages);
};

#endif /* BOOSTING_CLASSIFIERS_CASCADECLASSIFIER_H_ */
