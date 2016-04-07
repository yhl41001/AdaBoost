/*
 * utils.h
 *
 *  Created on: 16/mar/2016
 *      Author: lorenzocioni
 */

#ifndef BOOSTING_UTILS_UTILS_H_
#define BOOSTING_UTILS_UTILS_H_

enum example {POSITIVE, NEGATIVE};

//Sign function
template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

#endif /* BOOSTING_UTILS_UTILS_H_ */
