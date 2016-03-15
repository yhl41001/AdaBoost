/*
 * Main.cpp
 *
 *  Created on: 09/mar/2016
 *      Author: lorenzocioni
 */

#include <opencv2/core/core.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

int main( int argc, char** argv ){

	String filename = "/Applications/MAMP/htdocs/watss/calibration/calibExtr_1.yaml";
    Mat myMat;
    FileStorage fs(filename,FileStorage::READ);
    fs["Rotation"] >> myMat;

    cout << myMat << endl;

    return 0;
}
