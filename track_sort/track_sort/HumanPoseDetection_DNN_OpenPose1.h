#pragma once
#include<opencv2/dnn.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>

//#include<iostream>
//#include<chrono>
#include<random>
#include<set>
//#include<cmath>

#include <string>
#include <vector>
#include<opencv2\opencv.hpp>
#include<opencv2\dnn.hpp>
#include <iostream>
using namespace cv;
using namespace std;
using namespace cv::dnn;

////////////////////////////////
struct BodyKeyPoint
{
	BodyKeyPoint(cv::Point point, float probability)
	{
		this->id = -1;
		this->point = point;
		this->probability = probability;
	}

	int id;
	cv::Point point;
	float probability;
};
/*
std::ostream& operator << (std::ostream& os, const BodyKeyPoint& kp)
{
os << "Id:" << kp.id << ", Point:" << kp.point << ", Prob:" << kp.probability << std::endl;
return os;
}*/

////////////////////////////////
struct ValidPair
{
	ValidPair(int aId, int bId, float score)
	{
		this->aId = aId;
		this->bId = bId;
		this->score = score;
	}

	int aId;
	int bId;
	float score;
};
/*
std::ostream& operator << (std::ostream& os, const ValidPair& vp)
{
os << "A:" << vp.aId << ", B:" << vp.bId << ", score:" << vp.score << std::endl;
return os;
}*/

////////////////////////////////

template < class T > std::ostream& operator << (std::ostream& os, const std::vector<T>& v)
{
	os << "[";
	bool first = true;
	for (typename std::vector<T>::const_iterator ii = v.begin(); ii != v.end(); ++ii, first = false)
	{
		if (!first) os << ",";
		os << " " << *ii;
	}
	os << "]";
	return os;
}

template < class T > std::ostream& operator << (std::ostream& os, const std::set<T>& v)
{
	os << "[";
	bool first = true;
	for (typename std::set<T>::const_iterator ii = v.begin(); ii != v.end(); ++ii, first = false)
	{
		if (!first) os << ",";
		os << " " << *ii;
	}
	os << "]";
	return os;
}

////////////////////////////////

class HumanPoseDetection_DNN_OpenPose
{
public:

	HumanPoseDetection_DNN_OpenPose()
		: m_ConfidenceThreshold(.1)
	{}

	~HumanPoseDetection_DNN_OpenPose()
	{}

	// load model files
	// modelBinary - binary model file
	// modelConfiguration - proto network configuration file
	bool LoadTFModel(string modelBinary);

	// load from memory
	bool LoadTFModel(char *bufferModel, size_t modelSize, char *bufferProto = NULL, size_t protoSize = 0);

	// loaded network object
	// network object
	const dnn::Net& Network() const
	{
		return m_net;
	}

	// return number of detected pose
	int NumberOfPeople() const
	{
		return m_PersonwiseKeypoints.size();
	}

	// reset detection result
	void Clear()
	{
		m_PersonwiseKeypoints.clear();
	}

	// human pose detection
	bool DetectHumanPose(Mat& frame);

	// show detection points
	bool ShowPoseKeyPoints(Mat& frame);

private:

	// network object
	dnn::Net m_net;

	// dynamic data set:
	float m_ConfidenceThreshold;

	std::vector<std::vector<BodyKeyPoint>> m_DetectedKeypoints;
	std::vector<BodyKeyPoint> m_KeyPointsList;
	std::vector<std::vector<int>> m_PersonwiseKeypoints;
}; 
