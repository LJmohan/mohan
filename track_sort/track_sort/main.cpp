#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>

#include <vector>
//#include "SortTracker.h"
#include "StateSortTracker.h"
#include "time.h"
#include "HumanPoseDetection_DNN_OpenPose.h"
using namespace cv;
using namespace std;
using namespace cv::dnn;

const size_t inWidth = 512;
const size_t inHeight = 512;
const float inScaleFactor = 0.007843f;
const float meanVal = 127.5;
const char* classNames[] = { "background",
"person", "head", "spray", "walking",
"running", "falling", "jumping", "kicking", "punching",
"waving", "bending", "squat", "exercise",
"bed", "chair", "sofa", "desk" };

const String keys
= "{ help           | false | print usage         }"
"{ proto          | MobileNetSSD_deploy_swim.prototxt   | model configuration }"
"{ model          | swim-merge-2.3_iter_120000.caffemodel | model weights }"
"{ camera_device  | 0     | camera device number }"
"{ camera_width   | 640   | camera device width  }"
"{ camera_height  | 480   | camera device height }"
"{ video          | D://2023//1、溺水检测//溺水视频//swim.mp4 | video or image for detection}"
"{ out            | out.mp4      | path to output video file}"
"{ min_confidence | 0.5   | min confidence      }"
"{ opencl         | true | enable OpenCL }";

struct ObjRect
{
	float x1;
	float x2;
	float y1;
	float y2;
	int label;
};
struct spray_object
{
	Rect_<float> box;//记录水花位置
	int spray_num;//记录水花出现的次数
};
// Computes IOU between two bounding boxes
double GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt)
{
	float in = (bb_test & bb_gt).area();
	float un = bb_test.area() + bb_gt.area() - in;

	if (un < DBL_EPSILON)
		return 0;

	return (double)(in / un);
}
ofstream aa("1.txt");
ofstream bb("2.txt");

//int main(int argc, char** argv)
int main(int argc, char** argv)
{
	CommandLineParser parser(argc, argv, keys);
	parser.about("This sample uses MobileNet Single-Shot Detector "
		"(https://arxiv.org/abs/1704.04861) "
		"to detect objects on camera/video/image.\n"
		".caffemodel model's file is available here: "
		"https://github.com/chuanqi305/MobileNet-SSD\n"
		"Default network is 300x300 and 20-classes VOC.\n");

	if (parser.get<bool>("help"))
	{
		parser.printMessage();
		return 0;
	}

	clock_t start, finish;
	double duration;


	//String modelConfiguration = parser.get<String>("proto");
	//String modelBinary = parser.get<String>("model");
	String modelConfiguration = parser.get<String>("proto");
	String modelBinary = parser.get<String>("model");
	CV_Assert(!modelConfiguration.empty() && !modelBinary.empty());

	//! [Initialize network]
	dnn::Net net = readNetFromCaffe(modelConfiguration, modelBinary);
	//! [Initialize network]

	if (parser.get<bool>("opencl"))
	{
		net.setPreferableTarget(DNN_TARGET_OPENCL);
	}

	if (net.empty())
	{
		cerr << "Can't load network by using the following files: " << endl;
		cerr << "prototxt:   " << modelConfiguration << endl;
		cerr << "caffemodel: " << modelBinary << endl;
		cerr << "Models can be downloaded here:" << endl;
		cerr << "https://github.com/chuanqi305/MobileNet-SSD" << endl;
		exit(-1);
	}

	VideoCapture cap;
	if (!parser.has("video"))
	{
		int cameraDevice = parser.get<int>("camera_device");
		cap = VideoCapture(cameraDevice);
		if (!cap.isOpened())
		{
			cout << "Couldn't find camera: " << cameraDevice << endl;
			return -1;
		}

		cap.set(CAP_PROP_FRAME_WIDTH, parser.get<int>("camera_width"));
		cap.set(CAP_PROP_FRAME_HEIGHT, parser.get<int>("camera_height"));
	}
	else
	{
		cap.open(parser.get<String>("video"));
		if (!cap.isOpened())
		{
			cout << "Couldn't open image or video: " << parser.get<String>("video") << endl;
			return -1;
		}
	}

	//Acquire input size
	Size inVideoSize((int)cap.get(CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CAP_PROP_FRAME_HEIGHT));

	double fps = cap.get(CAP_PROP_FPS);
	int fourcc = static_cast<int>(cap.get(CAP_PROP_FOURCC));
	VideoWriter outputVideo;
	outputVideo.open(parser.get<String>("out"),
		(fourcc != 0 ? fourcc : VideoWriter::fourcc('M', 'J', 'P', 'G')),
		(fps != 0 ? fps : 10.0), inVideoSize, true);
	vector<vector<float>> DetectionRegionBlock;
	// sort tracking algrithm
	//SortTracker_Kalman m_ObjectTrackers;
	StateSortTracker_Kalman m_ObjectTrackers;
	// dnn detected objects (within user defined zones)
	//(useless) vector<DetectionRegionBlock> m_DNN_Detected_Objects;
	//openpose
	String weights = "MobileNetTFPose_432x368.pb";
	//LicensePlateRecognition_TF *openpose =new LicensePlateRecognition_TF();
	HumanPoseDetection_DNN_OpenPose *openpose = new HumanPoseDetection_DNN_OpenPose();
	openpose->LoadTFModel(weights);
	int n_frame = 0;
	
	vector<struct spray_object> spray_result;
	vector<Point> spray_head_num(100000);//spray_head_num.x表示水花区域被记录的次数，spray_head_num.y表示水花区域内出现head的次数
	for (;;)
	{
		aa << "spray_result.size(): " << spray_result.size() << endl;
		n_frame = n_frame + 1;
		Mat frame;
		cap >> frame; // get a new frame from camera/video or read image
		Mat frame_temp = frame.clone();
		if (frame.empty())
		{
			waitKey();
			break;
		}
		for (int j = 0; j < spray_result.size(); j++)
		{
			spray_head_num[j].x = spray_head_num[j].x + 1;

		}
		
		for (int j = 0; j < spray_result.size(); j++)
		{
			/*
			int max_x = 0;
			int min_x = 10000;
			int max_y = 0;
			int min_y = 10000;
			for (int m = 0; m < spray_result[j].size(); m++)
			{
				if (spray_result[j][m].x < min_x)
				{
					min_x = spray_result[j][m].x;
				}
				if ((spray_result[j][m].x+ spray_result[j][m].width) > max_x)
				{
					max_x = spray_result[j][m].x + spray_result[j][m].width;
				}
				if (spray_result[j][m].y < min_y)
				{
					min_y = spray_result[j][m].y;
				}
				if ((spray_result[j][m].y + spray_result[j][m].height) > max_y)
				{
					max_y = spray_result[j][m].y + spray_result[j][m].height;
				}
				//rectangle(frame_temp, Point(spray_result[j][m].x, spray_result[j][m].y), Point(spray_result[j][m].x + spray_result[j][m].width , spray_result[j][m].y + spray_result[j][m].height ), Scalar(0, 0, 255), 2);
			}*/
			
					
			rectangle(frame_temp, Point(spray_result[j].box.x, spray_result[j].box.y), Point(spray_result[j].box.x + spray_result[j].box.width, spray_result[j].box.y + spray_result[j].box.height), Scalar(0, 255, 0), 2);
			
			

		}
		
		if (frame.channels() == 4)
			cvtColor(frame, frame, COLOR_BGRA2BGR);
		//! [Prepare blob]
		Mat inputBlob = blobFromImage(frame, inScaleFactor,
			Size(inWidth, inHeight),
			Scalar(meanVal, meanVal, meanVal),
			false, false); //Convert Mat to batch of images
						   //! [Prepare blob]

						   //! [Set input blob]
		net.setInput(inputBlob); //set the network input
								 //! [Set input blob]
		start = clock();
		//! [Make forward pass]
		Mat detection = net.forward(); //compute output
									   //! [Make forward pass]
		finish = clock();
		duration = (double)(finish - start) / CLOCKS_PER_SEC;
		printf("%f seconds/n", duration);
		printf("\n");

		vector<double> layersTimings;
		double freq = getTickFrequency() / 1000;
		double time = net.getPerfProfile(layersTimings) / freq;

		Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

		if (!outputVideo.isOpened())
		{
			putText(frame, format("FPS: %.2f ; time: %.2f ms", 1000.f / time, time),
				Point(20, 20), 0, 0.5, Scalar(0, 0, 255));
		}
		//else
			//cout << "Inference time, ms: " << time << endl;

		float confidenceThreshold = parser.get<float>("min_confidence");
		//vector<Rect_<float>> boxes;
		vector<TrackedObject> boxes;
		vector<TrackedObject> boxes_new;
		vector<ObjRect>  detections;
		vector<ObjRect> detections_new;
		int num_spray = 0;
		for (int i = 0; i < detectionMat.rows; i++)
		{
			float confidence = detectionMat.at<float>(i, 2);
			
			if (confidence > confidenceThreshold)
			{
				vector<int> label_ID;
				size_t objectClass = (size_t)(detectionMat.at<float>(i, 1));

				int left = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
				int top = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
				int right = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
				int bottom = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);
				label_ID.push_back(objectClass);
				boxes.push_back(TrackedObject(Rect_<float>(left, top, abs(right - left), abs(bottom - top)), objectClass, vector<int>(1, objectClass)));
				//cout << boxes[0]._labelID.size() << endl;
				ObjRect rect;
				rect.x1 = left;
				rect.x2 = right;
				rect.y1 = top;
				rect.y2 = bottom;
				rect.label = objectClass;
				detections.push_back(rect);
				Rect_<float> a = Rect_<float>(left, top, abs(right - left), abs(bottom - top));
				//double rate= m_ObjectTrackers.GetIOU(a, a);
				int w = abs(right - left);
				int h = abs(bottom - top);
			    
				if (objectClass == 2)
				{
					int head_num = 0;
					for (int j = 0; j < spray_result.size(); j++)
					{
				        //计算当前人头和所有水花区域的中心点间的距离
						float dis = ((left + w / 2) - (spray_result[j].box.x + spray_result[j].box.width / 2))* (left + w / 2) - (spray_result[j].box.x + spray_result[j].box.width / 2);
						dis = dis + ((top + h / 2) - (spray_result[j].box.y + spray_result[j].box.height / 2)) * ((top + h / 2) - (spray_result[j].box.y + spray_result[j].box.height / 2));
						dis = sqrt(dis);

						aa << "dis: " << dis << endl;
						float in = (spray_result[j].box & a).area();
						float rate_head = (float)in / (float)a.area();
						if (rate_head > 0)
						{
							head_num = head_num + 1;
							spray_head_num[j] = Point(spray_head_num[j].x, spray_head_num[j].y + 1);
						}
						

					}
					
					rectangle(frame_temp, Point(left, top), Point(right, bottom), Scalar(220, 175, 0), 2);
					//rectangle(frame, Point(left, top), Point(right, bottom), Scalar(220, 175, 0), 2);
					//String label = format("%s ", classNames[objectClass]);
					String label = format("%s: %.2f", classNames[objectClass], confidence);
					int baseLine = 0;
					Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
					top = max(top, labelSize.height);
					rectangle(frame_temp, Point(left, top - labelSize.height),
						Point(left + labelSize.width, top + baseLine),
						Scalar(255, 255, 255), FILLED);
					//rectangle(frame, Point(left, bottom - labelSize.height),
					//Point(left + labelSize.width, bottom + baseLine),
					//Scalar(255, 255, 255), FILLED);
					putText(frame_temp, label, Point(left, top),
						0, 0.5, Scalar(0, 0, 255)); //FONT_HERSHEY_SIMPLEX
				}
				if (objectClass == 3)
				{
					num_spray = num_spray + 1;
					//aa << Point(left, top);
					//aa <<     Point(right, bottom) << endl;
					//cout << Point(left, top) << endl;
					//cout << Point(right, bottom)<< endl;
					if (spray_result.size() <= 0)
					{
						aa << Point(left, top);
						aa << Point(right, bottom) << endl;
						spray_object temp_a;
						temp_a.box = a;
						temp_a.spray_num = 1;
						spray_result.push_back(temp_a);
					}
					float rate_max = 0;
					int ID_spray = -1;
					for (int j = 0;j < spray_result.size(); j++)//水花组数
					{
						//rectangle(frame, Point(spray_result[j].x, spray_result[j].y), Point(spray_result[j].x+ spray_result[j].width/2, spray_result[j].y + spray_result[j].height / 2), Scalar(220, 175, 0), 2);
						float in = (spray_result[j].box& a).area();
						float rate = 0;
						if (spray_result[j].box.area() < a.area())
						{
							 rate = (float)in / (float)spray_result[j].box.area();
						}
						else
						{
							rate = (float)in / (float)a.area();
						}
						
						//cout << "rate: " << rate << endl;
						aa << rate << endl;
						if (rate > rate_max)
						{
							rate_max = rate;
							ID_spray = j;
							//spray_result.push_back(a);
							aa << Point(left, top);
							aa << Point(right, bottom) << endl;
						}	
						
					}
					if (rate_max > 0.3)
					{
						int max_x = 0;
						int min_x = 10000;
						int max_y = 0;
						int min_y = 10000;
						//for (int m = 0; m < spray_result[ID_spray].size(); m++)
						{
							if (spray_result[ID_spray].box.x < min_x)
							{
								min_x = spray_result[ID_spray].box.x;
							}
							if ((spray_result[ID_spray].box.x + spray_result[ID_spray].box.width) > max_x)
							{
								max_x = spray_result[ID_spray].box.x + spray_result[ID_spray].box.width;
							}
							if (spray_result[ID_spray].box.y < min_y)
							{
								min_y = spray_result[ID_spray].box.y;
							}
							if ((spray_result[ID_spray].box.y + spray_result[ID_spray].box.height) > max_y)
							{
								max_y = spray_result[ID_spray].box.y + spray_result[ID_spray].box.height;
							}
							//rectangle(frame_temp, Point(spray_result[j][m].x, spray_result[j][m].y), Point(spray_result[j][m].x + spray_result[j][m].width , spray_result[j][m].y + spray_result[j][m].height ), Scalar(0, 0, 255), 2);
						}
						//同组的水花直接合并，每组只有一个水花
						spray_result[ID_spray].box.x = min_x;
						spray_result[ID_spray].box.y = min_y;
						spray_result[ID_spray].box.width  = abs(max_x-min_x);
						spray_result[ID_spray].box.height = abs(max_y - min_y);
						spray_result[ID_spray].spray_num = spray_result[ID_spray].spray_num + 1;
						//spray_result[ID_spray].push_back(a);//加入已有的水花组
					}
					if (rate_max <= 0.3)
					{
						spray_object temp_a;
						temp_a.box = a;
						temp_a.spray_num = 1;
						spray_result.push_back(temp_a);
					}
					rectangle(frame, Point(left, top), Point(right, bottom), Scalar(220, 175, 0), 2);
					//String label = format("%s ", classNames[objectClass]);
					String label = format("%s: %.2f", classNames[objectClass], confidence);
					int baseLine = 0;
					Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
					top = max(top, labelSize.height);
					rectangle(frame, Point(left, top - labelSize.height),
						Point(left + labelSize.width, top + baseLine),
						Scalar(255, 255, 255), FILLED);
					//rectangle(frame, Point(left, bottom - labelSize.height),
					//Point(left + labelSize.width, bottom + baseLine),
					//Scalar(255, 255, 255), FILLED);
					putText(frame, label, Point(left, top),
						0, 0.5, Scalar(0, 0, 255)); //FONT_HERSHEY_SIMPLEX
				}
				
				
			}
			
		}
		//bb << "n_frame: " << n_frame << endl;
		bb <<"num_spray: "<< num_spray << endl;
		cout << "spray_head_num.size(): " << spray_head_num.size() << endl;
		for (int n = 0; n < spray_result.size(); n++)
		{
			//if (spray_head_num[n].y > 0)
			{
				putText(frame_temp, to_string(n), Point(spray_result[n].box.x , spray_result[n].box.y), FONT_HERSHEY_PLAIN,
					2, Scalar(0, 0, 255), 3); //FONT_HERSHEY_SIMPLEX
				cout << n<<"  spray_head_num[n].x : " << spray_head_num[n].x<<"  spray_head_num[n].y : " << spray_head_num[n].y <<"   spray_result[n].spray_num: "<< spray_result[n].spray_num<<   endl;
		    }
			if (spray_head_num[n].x > 100)
			{
				spray_head_num[n].x = 0;
				spray_head_num[n].y = 0;
				spray_result.erase(spray_result.begin() + n);
			}
			if (spray_head_num[n].y > 30)//&&spray_result[n].spray_num>2,水花检测到的次数比较少，如果设置水花出现的次数，会导致漏报
			{
				putText(frame_temp, "please help!!!", Point(150, 50), FONT_HERSHEY_PLAIN,
					4, Scalar(0, 0, 255),3); //FONT_HERSHEY_SIMPLEX
				putText(frame_temp, "warning", Point(spray_result[n].box.x, spray_result[n].box.y + spray_result[n].box.height / 2), FONT_HERSHEY_PLAIN,
					2, Scalar(0, 0, 255), 3); //FONT_HERSHEY_SIMPLEX
				rectangle(frame_temp, Point(spray_result[n].box.x, spray_result[n].box.y), Point(spray_result[n].box.x + spray_result[n].box.width, spray_result[n].box.y + spray_result[n].box.height), Scalar(0, 0, 255), 2);
				
			}
		}
		if (n_frame == 250)
		{
			//spray_result.clear();
		}
		//剔除错检bounding box
		double ratio = -1;
		double ratio_max = -1;
		int ID = -1;
		int ID_new = -1;
		int state = -1;
		vector<int> ID_number;
		vector<int> ID_result;
		
		int flag = 0;
		for (int i = 0; i < boxes.size(); i++)
		{
			vector<double> ID_match;//重复box的位置ID
			vector<int> ID_locationID;//重复box的locationID
			vector<int> ID_stateID;//重复box的stateID
			Rect_<float> rect1 = boxes[i]._bbox;
			for (int j = 0; j < boxes.size(); j++)
			{
				if (i != j)
				{
					Rect_<float> rect2 = boxes[j]._bbox;
					ratio = m_ObjectTrackers.GetIOU(rect1, rect2);
					
					if (ratio>ratio_max)
					{
						ratio_max = ratio;
						ID = j;
					}
					if (ratio_max > 0.75)
					{
						ID_match.push_back(ratio_max);
						ID_locationID.push_back(ID);
						ID_stateID.push_back(boxes[ID]._classID);
						ratio_max = -1;
						ID = -1;
					}
					
				}	
				
			}
			for (int m = 0; m < ID_match.size(); m++)
			{
				if (ID_match[m] > 0.75)
				{
					ID_number.push_back(ID_locationID[m]);
				}
			}
			/*if (ratio_max > 0.85)
			{
				ID_new = ID;
				//state = boxes[j]._classID;
			}
			if (ID_new >= 0)
			{			
				ID_number.push_back(ID_new);
			}*/
			
			if (ID_result.size() == 0)
			{
				for (int k = 0; k<ID_number.size(); k++)
				{
					if (i != ID_number[k])
					{
						flag = flag + 1;
					}
				}
			}
			else
			{
				for (int k = 0; k<ID_number.size(); k++)
				{
					for (int k1 = 0; k1 < ID_result.size(); k1++)
					{
						if (ID_result[k1] != ID_number[k] && i != ID_number[k])
						{
							flag = flag + 1;
						}
					}
				}
			}
			if (ID_result.size() == 0)
			{
				if (flag == ID_number.size())
				{
					ID_result.push_back(i);
					//detections_new.push_back(rect_1);				
					ID_stateID.push_back(boxes[i]._classID);
					boxes[i]._labelID.clear();
					boxes[i]._labelID = ID_stateID;
					boxes_new.push_back(boxes[i]);
					//cout << boxes[i]._classID << endl;
					if (ID_stateID.size() > 0)
					{
						//cout << ID_stateID[0] << endl;
					}
					
				}
			}
			else
			{
				if (flag == ID_number.size()*ID_result.size())
				{
					ID_result.push_back(i);
					ID_stateID.push_back(boxes[i]._classID);
					boxes[i]._labelID.clear();
					boxes[i]._labelID = ID_stateID;
					//detections_new.push_back(rect_1);
					boxes_new.push_back(boxes[i]);
					//cout << boxes[i]._classID << endl;
					if (ID_stateID.size() > 0)
					{
						//cout << ID_stateID[0] << endl;
					}
					
					//cout << ID_new << endl;
				}
			}
			ID_new = -1;
			ID = -1;
			ratio = -1;
			ratio_max = -1;
			state = -1;
			flag = 0;
			
		}
		/*
		double overlaparea = 0;
		double area1 = 0;
		double area2 = 0;
		double ratio = 0;
		double ratio_max = 0;
		int ID = -1;
		vector<int> ID_number;
		vector<int> ID_result;
		int flag = 0;
		for (int i = 0; i < detections.size(); i++)
		{
			
			ObjRect rect_1 = detections[i];
			TrackedObject trackbox;
			trackbox._bbox = Rect_<float>(rect_1.x1, rect_1.y1, abs(rect_1.x1 - rect_1.x2), abs(rect_1.y1 - rect_1.y2));
			trackbox._classID = rect_1.label;
			
			for (int j = 0; j < detections.size(); j++)
			{
				ObjRect rect_2 = detections[j];
				if (i != j)
				{
					//计算当前帧所有矩形框的重叠面积              
					double endX = MAX(rect_1.x2, rect_2.x2);
					double startX = MIN(rect_1.x1, rect_2.x1);
					double width = (rect_1.x2 - rect_1.x1) + (rect_2.x2 - rect_2.x1) - (endX - startX);

					double endY = MAX(rect_1.y2, rect_2.y2);
					double startY = MIN(rect_1.y1, rect_2.y1);
					double height = (rect_1.y2 - rect_1.y1) + (rect_2.y2 - rect_2.y1) - (endY - startY);

					//计算矩形框重叠率
					if (width <= 0 || height <= 0)
					{
						overlaparea = 0;
						ratio = 0;
					}
					else
					{
						area1 = (rect_1.y2 - rect_1.y1)*(rect_1.x2 - rect_1.x1);
						area2 = (rect_2.y2 - rect_2.y1)*(rect_2.x2 - rect_2.x1);
						overlaparea = width*height;
						if ((area1 + area2 - overlaparea)>0)
						{
							ratio = overlaparea / (area1 + area2 - overlaparea);
						}
					}
					if (ratio>ratio_max)
					{
						ratio_max = ratio;
						ID = j;
					}
					
				}

			}
			
			if (ID >= 0)
			{
				ID_number.push_back(ID);
			}
			ID = -1;
			if (ID_result.size() == 0)
			{
				for (int k = 0; k<ID_number.size(); k++)
				{
					if (i != ID_number[k])
					{
						flag = flag + 1;
					}
				}
			}
			else
			{
				for (int k = 0; k<ID_number.size(); k++)
				{
					for (int k1 = 0; k1 < ID_result.size(); k1++)
					{
						if (ID_result[k1] != ID_number[k]&&i!= ID_number[k])
						{
							flag = flag + 1;
						}
					}
				}
			}
			if (ID_result.size() == 0)
			{
				if (flag == ID_number.size())
				{
					ID_result.push_back(i);
					detections_new.push_back(rect_1);
					boxes_new.push_back(trackbox);
				}
			}
			else
			{
				if (flag == ID_number.size()*ID_result.size())
				{
					ID_result.push_back(i);
					detections_new.push_back(rect_1);
					boxes_new.push_back(trackbox);
				}
			}
			flag = 0;
			ratio_max = 0;
			ratio = 0;
		}
		*/

		if (!boxes_new.empty())
		{
			//cout << boxes_new[0]._labelID.size() << endl;
			m_ObjectTrackers.Tracking(boxes_new);
		}
		int num = 0;
		for (int i = 0; i < boxes_new.size(); i++)
		{
			if (boxes_new[i]._classID != 14 && boxes_new[i]._classID != 15 && boxes_new[i]._classID != 16 && boxes_new[i]._classID != 17)
			{
				num = num + 1;
			}
		}
		m_ObjectTrackers.people_num = num;
		// draw object trace of tracking
		m_ObjectTrackers.Show(frame);
		if (n_frame == 250)
		{
			n_frame = 0;
		}
		//if (n_frame == 1)
		{
			//openpose->DetectHumanPose(frame);
			//openpose->ShowPoseKeyPoints(frame);
		}
		
		// write video
		if (outputVideo.isOpened())
			outputVideo << frame;
		//namedWindow("detection_hht", WINDOW_NORMAL);
		imshow("detections_hht", frame);

		imshow("frame_temp", frame_temp);

		if (waitKey(1) >= 0) break;
	}

	return 0;
} // main