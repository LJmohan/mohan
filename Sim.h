#ifndef SIM_H
#define SIM_H
#define pi 3.14;
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;   
using namespace cv; 


class Sim
{
public:
	Sim():
	  W(4),
      NBINS(9),
	  NBP(5),
	  step(20),
	  SIM_OK(0)
	{
	  top = W; 
      bottom =W;
      left = W; 
      right = W;	 
	 }

	~Sim()
	{}
private:
	int W ; 
    Mat dst;
	int NBINS;//each bin for 20 degree
	int NBP;
	int step;//steps
	int top ; 
    int bottom;
    int left; 
    int right;  
	Mat magnitudes;
    Mat angles;
    Mat sobelx;  
    Mat sobely; 
	vector<vector<Point> > contours;
	float theta;//Ԫ�ط���Ƕ�
    float theta1;//��������Ƕ�
    float mag;
	vector<float> descriptor;
    vector<vector<float>> V;
	int SIM_OK;//���巵��ֵ
  
public:

	//1.�����ݶ�ģ
	int gradient(const Mat& src,Mat& magnitudes);//�������� &magnitudes

	//2.�����ݶȷ����
	int angle(const Mat& src,Mat& angles);//�������� &angles

	//3.ͳ���ݶȷ���ֲ�ֱ��ͼ
	//vector<float>& HistoGram(const Mat& src,const Mat& magnitudes, const Mat& angles, int rstrong,int cstrong,vector<float>& histo);
	int HistoGram(const Mat& src,const Mat& magnitudes, const Mat& angles, int rstrong,int cstrong,vector<float>& histo);//��ʽ�ط������ã�������ʾ����

	//4.����������
	float orientation(vector<float> histo);

	//5.������������
	//vector<float>& FeatureVector(const Mat& src,const Mat& magnitudes, const Mat& angles,int rstrong, int cstrong,float theta,vector<float>& descriptor1);
	int FeatureVector(const Mat& src,const Mat& magnitudes, const Mat& angles,int rstrong, int cstrong,float theta,vector<float>& descriptor1);

	//6.���㷽��ֱ��������
	vector<float> bin()
	{
	 vector<float> bin(NBINS);
	 bin[0]=step/2;
     for(int i=1;i<NBINS;i++)
	  {
	    bin[i]=bin[i-1]+step;
      } 
	 return bin;
	};
		
};

#endif
