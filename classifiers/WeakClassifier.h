/*
 * Classifier.h
 *
 *  Created on: 15/mar/2016
 *      Author: lorenzocioni
 */

#ifndef CLASSIFIERS_WEAKCLASSIFIER_H_
#define CLASSIFIERS_WEAKCLASSIFIER_H_

class WeakClassifier {

private:
	double error;


public:
	void evaluate();


	double getError() const;
	void setError(double error);
};


#endif /* CLASSIFIERS_WEAKCLASSIFIER_H_ */
