/*
 * MultiClassClassifier.h
 *
 *  Created on: 18/mag/2016
 *      Author: lorenzocioni
 */

#ifndef BOOSTING_CLASSIFIERS_MULTICLASSCLASSIFIER_H_
#define BOOSTING_CLASSIFIERS_MULTICLASSCLASSIFIER_H_
#include <vector>
#include <iostream>
#include <cmath>
#include "StrongClassifier.h"
#include "MultiWeakClassifier.h"

using namespace std;

class MultiClassClassifier: public StrongClassifier {
private:
	vector<MultiWeakClassifier> multiClassClassifiers;

public:
	MultiClassClassifier();
	int predict(Data* x);
	~MultiClassClassifier();
	const vector<MultiWeakClassifier>& getMultiClassClassifiers() const;
	void setMultiClassClassifiers(
			const vector<MultiWeakClassifier>& multiClassClassifiers);
};


#endif /* BOOSTING_CLASSIFIERS_MULTICLASSCLASSIFIER_H_ */
