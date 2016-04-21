/*
 * Utils.hpp
 *
 *  Created on: 11/apr/2016
 *      Author: lorenzocioni
 */

#ifndef FACEDETECTOR_UTILS_UTILS_HPP_
#define FACEDETECTOR_UTILS_UTILS_HPP_

#include <dirent.h>
#include <string>
#include <vector>

class Utils {
public:
	static std::vector<std::string> open(std::string path = ".") {
	    DIR*    dir;
	    dirent* pdir;
	    std::vector<std::string> files;

	    dir = opendir(path.c_str());

	    while (pdir = readdir(dir)) {
	    	if(strcmp( pdir->d_name, ".") != 0 && strcmp(pdir->d_name, "..") != 0 )
	    		files.push_back(pdir->d_name);
	    }

	    return files;
	}

	/**
	 * Merge a set of rectangles if there's an overlap between each rectangle for more than
	 * specified overlap area
	 * @param   boxes a set of rectangles to be merged
	 * @param   overlap the minimum area of overlap before 2 rectangles are merged
	 * @param   group_threshold only the rectangles that have more than the remaining group_threshold rectangles will be retained
	 * @return  a set of merged rectangles
	 **/
	static vector<Rect> mergeRectangles( const vector<Rect>& boxes, float overlap, int group_threshold ) {
	    vector<Rect> output;
	    vector<Rect> intersected;
	    vector< vector<Rect> > partitions;
	    vector<Rect> rects( boxes.begin(), boxes.end() );

	    while( rects.size() > 0 ) {
	        Rect a      = rects[rects.size() - 1];
	        int a_area  = a.area();
	        rects.pop_back();

	        if( partitions.empty() ) {
	            vector<Rect> vec;
	            vec.push_back( a );
	            partitions.push_back( vec );
	        }
	        else {
	            bool merge = false;
	            for( int i = 0; i < partitions.size(); i++ ){

	                for( int j = 0; j < partitions[i].size(); j++ ) {
	                    Rect b = partitions[i][j];
	                    int b_area = b.area();

	                    Rect intersect = a & b;
	                    int intersect_area = intersect.area();

	                    if (( a_area == b_area ) && ( intersect_area >= overlap * a_area  ))
	                        merge = true;
	                    else if (( a_area < b_area ) && ( intersect_area >= overlap * a_area  ) )
	                        merge = true;
	                    else if (( b_area < a_area ) && ( intersect_area >= overlap * b_area  ) )
	                        merge = true;

	                    if( merge )
	                        break;
	                }

	                if( merge ) {
	                    partitions[i].push_back( a );
	                    break;
	                }
	            }

	            if( !merge ) {
	                vector<Rect> vec;
	                vec.push_back( a );
	                partitions.push_back( vec );
	            }
	        }
	    }

	    for( int i = 0; i < partitions.size(); i++ ) {
	        if( partitions[i].size() <= group_threshold )
	            continue;

	        Rect merged = partitions[i][0];
	        for( int j = 1; j < partitions[i].size(); j++ ) {
	            merged |= partitions[i][j];
	        }

	        output.push_back( merged );

	    }

	    return output;
	}
};



#endif /* FACEDETECTOR_UTILS_UTILS_HPP_ */
