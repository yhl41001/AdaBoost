/*
 * Classifier.h
 *
 *  Created on: 15/mar/2016
 *      Author: lorenzocioni
 */

#ifndef CLASSIFIERS_CLASSIFIER_H_
#define CLASSIFIERS_CLASSIFIER_H_

class Classifier {

private:
	double error;


public:
	double getError() const;
	void setError(double error);
};


#endif /* CLASSIFIERS_CLASSIFIER_H_ */
