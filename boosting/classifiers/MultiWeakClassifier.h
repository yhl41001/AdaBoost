/*
 * MultiWeakClassifier.h
 *
 *  Created on: 21/mag/2016
 *      Author: lorenzocioni
 */

#ifndef BOOSTING_CLASSIFIERS_MULTIWEAKCLASSIFIER_H_
#define BOOSTING_CLASSIFIERS_MULTIWEAKCLASSIFIER_H_

#include <vector>
#include "WeakClassifier.h"

using namespace std;

class MultiWeakClassifier : public WeakClassifier {
private:
	int classes;

public:
	MultiWeakClassifier(int classes);
	int predict(Data* x);
	int predict(vector<double> x);
	~MultiWeakClassifier();
};



#endif /* BOOSTING_CLASSIFIERS_MULTIWEAKCLASSIFIER_H_ */
