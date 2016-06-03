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
	this->trainImages = {};
	this->trainLabels = {};
	this->scales = scales;
	this->detectionWindowSize = 24;
	this->delta = 2;
	this->stages = 24;
	cout << "  -Scales: " << scales << "\n  -Window size: "<< detectionWindowSize << endl;
	boost = new ViolaJones(trainedCascade);
}

FaceDetector::FaceDetector(vector<Mat> trainImages, vector<int> trainLabels, int stages, int detectionWindowSize){
	cout << "FaceDetector\n************" << endl;
	this->trainImages = trainImages;
	this->trainLabels = trainLabels;
	this->scales = 12;
	this->delta = 2;
	this->stages = stages;
	this->detectionWindowSize = detectionWindowSize;
	boost = new ViolaJones();
	cout << "  -Scales: " << scales << "\n  -Window size: "<< detectionWindowSize << endl;

}

void FaceDetector::train(){
	vector<Data*> positives;
	vector<Data*> negatives;
	double percent = 0;
	auto t_start = chrono::high_resolution_clock::now();

	cout << "\nExtracting image features" << endl;

	for(int i = 0; i < trainImages.size(); ++i){
		Mat intImg = IntegralImage::computeIntegralImage(trainImages[i]);
		//Extracting haar like features
		vector<double> features = HaarFeatures::extractFeatures(intImg, detectionWindowSize, 0, 0);
		/*	Initialize weights */
		if(trainLabels[i] == 1){
			positives.push_back(new Data(features, trainLabels[i]));
		} else {
			negatives.push_back(new Data(features, trainLabels[i]));
		}
		percent = (double) i * 100 / (trainImages.size() - 1) ;
		cout << "\rEvaluated: " << i + 1 << "/" << trainImages.size() << " images" << flush;
	}

	cout << "\nExtracted features in ";
	auto t_end = chrono::high_resolution_clock::now();
	cout << std::fixed << (chrono::duration<double, milli>(t_end - t_start).count())/1000 << " s\n";

	boost = new ViolaJones(positives, negatives, stages);
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
	int thickness = 1;

	//For each image scale
	for(int s = 0; s < scales; ++s){
		scaleRefactor = pow(scaleFactor, s);
		cout << "Scale factor " << scaleRefactor << endl;

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

	cout << "Detected: " << predictions.size() << " faces" << endl;
    //predictions = boost->mergeDetections(predictions);
    cout << "Merged into: " << predictions.size() << " faces" << endl;
	if(showResults){
		double norm;
		for_each(predictions.begin(), predictions.end(), [&norm] (const Face& face) {
			norm += face.getScore();
		});
		for(unsigned int i = 0; i < predictions.size(); ++i){
			if(showScores){
				string text = to_string(predictions[i].getScore());
				Point textOrg(predictions[i].getRect().x + 5,
						predictions[i].getRect().y + 15);
				putText(img, text, textOrg, fontFace, fontScale,
							        Scalar::all(255), thickness, 8);
			}
			if(predictions[i].getScore() / norm > 0.5){
				thickness = 2;
			}
			rectangle(img, predictions[i].getRect(), Scalar::all(255), thickness);
		}
		imshow("img", img);
		waitKey(0);
		imwrite("out.jpg", img);
	}

	return predictions;
}

void FaceDetector::displaySelectedFeatures(Mat img){
	resize(img, img, Size(24, 24));
}

FaceDetector::~FaceDetector(){
	trainImages.clear();
	trainLabels.clear();
}
