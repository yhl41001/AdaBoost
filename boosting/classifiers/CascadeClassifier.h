/*
 * CascadeClassifier.hpp
 *
 *  Created on: 11/apr/2016
 *      Author: lorenzocioni
 */

#ifndef BOOSTING_CLASSIFIERS_CASCADECLASSIFIER_H_
#define BOOSTING_CLASSIFIERS_CASCADECLASSIFIER_H_

class CascadeClassifier {
private:
	int stages;
	int number;

public:
	CascadeClassifier(int stages);
	void train();
	~CascadeClassifier();
};



#endif /* BOOSTING_CLASSIFIERS_CASCADECLASSIFIER_H_ */
