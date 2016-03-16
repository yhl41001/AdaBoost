/*
 * StrongClassifier.h
 *
 *  Created on: 16/mar/2016
 *      Author: lorenzocioni
 */

#ifndef CLASSIFIERS_STRONGCLASSIFIER_H_
#define CLASSIFIERS_STRONGCLASSIFIER_H_

#include <vector>
#include "WeakClassifier.h"
#include "../features/Feature.h"

class StrongClassifier {

private:
	vector<WeakClassifier> classifiers;

public:
	StrongClassifier();
	int predict(Feature x);
	~StrongClassifier();
};



#endif /* CLASSIFIERS_STRONGCLASSIFIER_H_ */
