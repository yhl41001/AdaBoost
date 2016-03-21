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
#include "../features/Data.h"

class StrongClassifier {

private:
	vector<WeakClassifier> classifiers;
	bool trained;

public:
	StrongClassifier(vector<WeakClassifier> classifiers);
	int predict(Data x);
	~StrongClassifier();
	const vector<WeakClassifier>& getClassifiers() const;
	void setClassifiers(const vector<WeakClassifier>& classifiers);
	bool isTrained() const;
	void setTrained(bool trained);
};



#endif /* CLASSIFIERS_STRONGCLASSIFIER_H_ */
