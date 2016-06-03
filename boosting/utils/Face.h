/*
 * Face.h
 *
 *  Created on: 27/mag/2016
 *      Author: lorenzocioni
 */

#ifndef BOOSTING_UTILS_FACE_H_
#define BOOSTING_UTILS_FACE_H_

#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

class Face {

private:
	Rect rect;
	double score;

public:
	Face(Rect rect);
	~Face();
	const Rect& getRect() const;
	void setRect(const Rect& rect);
	double getScore() const;
	void setScore(double score);
};



#endif /* BOOSTING_UTILS_FACE_H_ */
