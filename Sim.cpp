#include "Sim.h" 
qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqdsddddddddddddddddddddddfsfsf
//1.计算梯度模
hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh'kloy7ir6udrthrtyhty
//11111111111111111111111111111
int Sim::gradient(const Mat& src,Mat& magnitudes)
{
dst = src;
const  CvArr* dst1=(CvArr*)&src;
int borderType=BORDER_REPLICATE;
copyMakeBorder( src, dst, top, bottom, left, right, borderType);
Sobel(dst, sobelx, CV_32F, 1, 0, 3); //sobel算子求梯度 
Sobel(dst, sobely, CV_32F, 0, 1, 3);
magnitude(sobelx,sobely,magnitudes);

return SIM_OK;
}

//2.计算梯度方向角
int Sim::angle(const Mat& src,Mat& angles)
{
phase(sobelx,sobely,angles,false);
return SIM_OK;

}

//3.统计梯度方向分布直方图
int Sim::HistoGram(const Mat& src,const Mat& magnitudes, const Mat& angles, int rstrong, int cstrong,vector<float>& histo)
{
rstrong=rstrong+W;
cstrong=cstrong+W;
rstrong=floor(rstrong+0.5);
cstrong=floor(cstrong+0.5);
int X1=rstrong-W;
int X2=rstrong+W;
int Y1=cstrong-W;
int Y2=cstrong+W;

vector<float> bin=Sim::bin();

for ( int x =X1; x <=X2; x++)
{
	if(x>=0 && x<dst.rows) 
		 {	            
		       for(int y =Y1; y <=Y2; y++)
			     {
	                   if(y>=0 && y<dst.cols)
				 
				      {
			            theta=(float)angles.at<float>(x,y)*180/pi;
						mag=(float)magnitudes.at<float>(x,y);
		
                        if (theta>180)
						{
                          theta1=theta-180;
						}
						else
						{
						  theta1=theta;
						}
		                
                        for(int k=0; k<NBINS-1;k++ )
						{			   
                           if(theta1>bin[k]&&theta1<bin[k+1])
				             {
                               int indx=k;
                               float w1=(bin[indx+1]-theta1)/step*mag;//for indx
                               float w2=(theta1-bin[indx])/step*mag;//for indx+1
				               histo[indx]=histo[indx]+w1;
				               histo[indx+1]=histo[indx]+w2; 
				             }
						}
			                
                        if(theta1<bin[0])
			            {
                           int indx=0;
                           float w1=(step-theta1)/step*mag;//for indx
                           float w2=(bin[indx]-theta1)/step*mag;//for indx+1
				           histo[indx]=histo[indx]+w1;
				           histo[NBINS-1]=histo[NBINS-1]+w2;
			            }
         
                       else if(theta1>bin[NBINS-1])
			            {
                           int indx=NBINS-1;
			               float w1=(bin[indx]+step-theta1)/step*mag;//for indx
                           float w2=(theta1-bin[indx])/step*mag;//for indx+1
				           histo[indx]=histo[indx]+w1;
				           histo[0]=histo[0]+w2;
                     
			            }
			  			 
					  
					 }
			   }
	     }
}
return SIM_OK;
}


//4.为边缘特征点分配主方向
float Sim::orientation(vector<float> histo)
{
	 vector<float> bin=Sim::bin();
     double minVal=0,maxVal=0;  
	 cv::Point minPt, maxPt; 
     minMaxLoc(histo,&minVal,&maxVal,&minPt,&maxPt);  
     double theta_max = maxVal;	
	 for (int ind=0; ind<NBINS; ind++)
     {
       if(histo[ind] == theta_max)
	      {
            int theta_indx =ind;
            theta1= bin[theta_indx]+step/2;
		    return theta1;				
		  }  
    }


}
	

//5.计算特征向量
int  Sim::FeatureVector(const Mat& src, const Mat& magnitudes, const Mat& angles,int rstrong, int cstrong,float theta,vector<float>& descriptor1)
{
rstrong=rstrong+W;
cstrong=cstrong+W;
rstrong=floor(rstrong+0.5);
cstrong=floor(cstrong+0.5);

vector<Point2f> points(NBP);
points[0]=Point2f(rstrong-W,cstrong);
points[1]=Point2f(rstrong+W,cstrong);
points[2]=Point2f(rstrong,cstrong);
points[3]=Point2f(rstrong,cstrong-W);
points[4]=Point2f(rstrong,cstrong+W);

vector<float> descriptor;
for (size_t i=0;i<points.size();i++)
  {

	points[i].x=points[i].x+W;
	points[i].y=points[i].y+W;
	points[i].x=floor(points[i].x+0.5);
	points[i].y=floor(points[i].y+0.5);

	int X1=points[i].x-W;
	int X2=points[i].x+W;
	int Y1=points[i].y-W;
	int Y2=points[i].y-W;

	vector<float> histo(NBINS);
	vector<float> bin=Sim::bin();

   for ( int x =X1; x <=X2; x++)
   {
	     if(x>=0 && x<dst.rows) 
		   {	            
		       for(int y =Y1; y <=Y2; y++)
			     {
	                   if(y>=0 && y<dst.cols)
				 
				      {
			            float angle=(float)angles.at<float>(x,y);
						double PI=2*pi;
						double theta1=-angle + theta;
						float angle1 = fmod(theta1,PI);   
						float theta2=angle1*180/pi;
			
						float theta3;
                        if (theta2>180)
						{
                           theta3=theta2-180;
						}
						else
						{
							theta3=theta2;
						}
				                
                        for(int k=1; k<NBINS-1;k++ )
						{			   
                          if(theta3>bin[k]&&theta3<bin[k+1])
				            {
                               int indx=k;
                               float w1=(bin[indx+1]-theta3)/step*mag;//for indx
                               float w2=(theta3-bin[indx])/step*mag;//for indx+1
				               histo[indx]=histo[indx]+w1;
				               histo[indx+1]=histo[indx]+w2;
				            }
							  
						}
			               
                        if(theta3<bin[0])
			            {
                           int indx=0;
                           float w1=(step-theta3)/step*mag;//for indx
                           float w2=(bin[indx]-theta3)/step*mag;//for indx+1
				           histo[indx]=histo[indx]+w1;
				           histo[NBINS-1]=histo[NBINS-1]+w2;
			            }
         
                       else if(theta3>bin[NBINS-1])
			            {
                           int indx=NBINS-1;
			               float w1=(bin[indx]+step-theta3)/step*mag;//for indx
                           float w2=(theta3-bin[indx])/step*mag;//for indx+1
				           histo[indx]=histo[indx]+w1;
				           histo[0]=histo[0]+w2;
                     
			            }
						
					 }
			   }
	      }
       }

	for (int j=0;j<NBINS;j++)
	{
      descriptor.push_back(histo[j]);
							
	}
			
  }

double NORM=norm(descriptor,NORM_L2)+1e-10;
for (int j=0;j<NBP*NBINS;j++)
	{
       descriptor1[j]=descriptor[j]/NORM;
	}
return SIM_OK;

}
					
			
