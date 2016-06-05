/*
 * FaceDetector.cpp
 *
 *  Created on: 07/apr/2016
 *      Author: lorenzocioni
 *
 *  Face detector implementing Viola&Jones algorithm
 *  AdaBoost extension for real time face detection using cascade classifiers
 */

#include "FaceDetector.h"

FaceDetector::FaceDetector(string trainedCascade, int scales){
	cout << "FaceDetector\n************" << endl;
	this->scales = scales;
	this->detectionWindowSize = 24;
	this->delta = 2;
	this->stages = 24;
	this->numNegatives = 0;
	this->numPositives = 0;
	cout << "  -Scales: " << scales << "\n  -Window size: "<< detectionWindowSize << endl;
	boost = new ViolaJones(trainedCascade);
}

FaceDetector::FaceDetector(string positivePath, string negativePath, int stages, int numPositives, int numNegatives, int detectionWindowSize){
	cout << "FaceDetector\n************" << endl;
	this->positivePath = positivePath;
	this->negativePath = negativePath;
	this->scales = 12;
	this->delta = 2;
	this->stages = stages;
	this->detectionWindowSize = detectionWindowSize;
	this->numNegatives = numNegatives;
	this->numPositives = numPositives;
	boost = new ViolaJones();
	cout << "  -Scales: " << scales << "\n  -Window size: "<< detectionWindowSize << endl;

}

void FaceDetector::setValidationPath(string validationPath){
	this->validationPath = validationPath;
}

void FaceDetector::train(){
	boost = new ViolaJones(positivePath, negativePath, stages, numPositives, numNegatives, detectionWindowSize);
	if(validationPath != ""){
		boost->setValidationPath(validationPath);
	}
	boost->train();
}

vector<Face> FaceDetector::detect(Mat img, bool showResults, bool showScores){
	vector<Face> predictions;
	vector<double> features;
	double scaleFactor = 0.75;
	double scaleRefactor;
	int prediction = 0;
	Mat tmp = img;
	Mat dst, window, intImg, det;
	int x, y, w;
	int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
	double fontScale = 0.5;

	//Evaluating computation time
	auto t_start = chrono::high_resolution_clock::now();

	//For each image scale
	for(int s = 0; s < scales; ++s){
		scaleRefactor = pow(scaleFactor, s);
		//Detection window slides
		intImg = IntegralImage::computeIntegralImage(tmp);
		for(int j = 0; j < tmp.rows - detectionWindowSize - delta; j += delta){
			for(int i = 0; i < tmp.cols - detectionWindowSize - delta; i += delta){
				window = intImg(Rect(i, j, detectionWindowSize, detectionWindowSize));
				prediction = boost->predict(window);
				if(prediction == 1) {
					x = (int) i / scaleRefactor;
					y = (int) j / scaleRefactor;
					w = (int) detectionWindowSize / scaleRefactor;
					if(w * w > 225){
						predictions.push_back(Face(Rect(x, y, w, w)));
					}
				}

			}
		}

		resize(tmp, dst, Size(), scaleFactor, scaleFactor);
		dst.copyTo(tmp);
	}

    predictions = boost->mergeDetections(predictions);
    cout << "Detected " << predictions.size() << " faces" << endl;

	auto t_end = high_resolution_clock::now();
	cout << "Time: " << (duration<double, milli>(t_end - t_start).count())/1000 << " s" << endl;

	if(showResults){
		for(unsigned int i = 0; i < predictions.size(); ++i){
			if(showScores){
				string text = to_string(predictions[i].getScore());
				Point textOrg(predictions[i].getRect().x + 5,
						predictions[i].getRect().y + 15);
				putText(img, text, textOrg, fontFace, fontScale,
							        Scalar::all(255), 1, 8);
			}
			rectangle(img, predictions[i].getRect(), Scalar::all(255));
		}
		imshow("img", img);
		waitKey(0);
		imwrite("out.jpg", img);
	}

	return predictions;
}

void FaceDetector::displaySelectedFeatures(Mat img, int index){
	resize(img, img, Size(detectionWindowSize, detectionWindowSize));
	vector<Stage*> stages = boost->getClassifier().getStages();
	int count = 0;
	for(unsigned int i = 0; i < stages.size(); ++i){
		for(unsigned int j = 0; j < stages[i]->getClassifiers().size(); ++j){
			if(count == index){
				for(unsigned int w = 0; w < stages[i]->getClassifiers()[j].getWhites().size(); ++w){
					rectangle(img, stages[i]->getClassifiers()[j].getWhites()[w], Scalar::all(255), CV_FILLED);
				}
				for(unsigned int b = 0; b < stages[i]->getClassifiers()[j].getBlacks().size(); ++b){
					rectangle(img, stages[i]->getClassifiers()[j].getBlacks()[b], Scalar::all(0), CV_FILLED);
				}
				imshow("feature", img);
				waitKey(0);
				imwrite("feature.jpg", img);
				return;
			}
			count++;
		}
	}
}

FaceDetector::~FaceDetector(){}
