#include "Sim.h" 

int main()  
{ 
 Sim ss;
 Mat src0= imread("cur53.jpg",1);
 Mat src=src0(Rect(400,800,300,350));
 imshow("原始图像", src);

 //canny detection
 int threshold=100;
 Mat src1;
 Canny( src, src1,threshold,threshold*2,3);
 imshow("canny边缘图像", src1);

 Mat magnitudes;
 ss.gradient(src,magnitudes);
 Mat angles;
 ss.angle(src,angles);

 //find  Contours
 vector<vector<Point> > contours;
 vector<Vec4i>hierarchy;
 findContours(src1, contours,hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
 int A,B;
 A=contours.size(); 
 vector<float>Num(A);
 vector<float>seed(250000);
 vector<Point2f> seed1(250000);

 //parameters 
 int t1=0;
 int t2=0;
 int tt=0;
 int st=2;
 float thre_H=0.7;
 float thre_L=0.3;
 float wloc=0.8;
 float wglo=0.2;
 int mini_SegLength=10;
 int NBINS=9;//each bin for 20 degree
 int NBP=5;
 int DIM=NBINS*NBP;
 for (int k1=0;k1<A;k1++)
	
 {
	 B=contours[k1].size();
	 vector<vector<float>> V(B);
	 for (int k2=0;k2<B;k2++)
	    {  
	      int XX=contours[k1][k2].x;
	      int YY=contours[k1][k2].y;
		  vector<float> histo_init(NBINS,0);
		  vector<float>& histo=histo_init;//引用初始化
		  ss.HistoGram(src, magnitudes, angles, XX, YY,histo);  //统计梯度方向分布直方图
          float theta=ss.orientation( histo); //分配主方向
		  vector<float> descriptor_init(DIM,0);
		  vector<float> descriptor1=descriptor_init;//引用初始化
	      ss.FeatureVector(src,magnitudes, angles,XX, YY,theta,descriptor1);  //计算特征向量
		  V[k2]=descriptor1;
	    }
	 
		 if(B>4)
       {   
	     double s1,s2,s3,s4,s5;
		 double ss=0;
		 vector<float> locsim(B-4);
		 vector<float> glosim(B-4);
		 vector<float> selfsim(B-4);
		 vector<float> nearsim(B-4);
		      //计算自相似性值
		 for ( int k2=2;k2<=B-3;k2++)
		 {

			 //局部自相似性值
			 CvMat V1,V2,V3,V4,V5;
			 double v1[45],v2[45],v3[45],v4[45],v5[45];
			 for(int j=0;j<DIM;j++)
	         {
				 v1[j]= V[k2][j];
				 v2[j]= V[k2-1][j];
				 v3[j]= V[k2-2][j];
                 v4[j]= V[k2+1][j];
				 v5[j]= V[k2+2][j];
			  }
			 V1=cvMat(DIM, 1, CV_64FC1, v1);
             V2=cvMat(DIM, 1, CV_64FC1, v2);
             V3=cvMat(DIM, 1, CV_64FC1, v3);
		     V4=cvMat(DIM, 1, CV_64FC1, v4);
             V5=cvMat(DIM, 1, CV_64FC1, v5);

             s1=cvDotProduct(&V1,&V1); 
		     s2=cvDotProduct(&V1,&V2); 
		     s3=cvDotProduct(&V1,&V3); 
			 s4=cvDotProduct(&V1,&V4); 
			 s5=cvDotProduct(&V1,&V5); 

			 locsim[k2-2]=(s2+s3+s4+s5)/4;
			 nearsim[k2-2]=s4;

			 //整体自相似性值	 
			 for ( int jj=2;jj<=B-3;jj++)
			 {
				if(jj!=k2)
				{
				  for(int j=0;j<DIM;j++)
	               {		    
		              ss=ss+V[k2][j]*V[jj][j];
		           }
				 }
			 }		   
		      glosim[k2-2]=ss/(B-4);
			  ss=0;
							
			  //最终自相似性值	
			  selfsim[k2-2]=0.8*locsim[k2-2]+ 0.2*glosim[k2-2];
			  //cout<<selfsim[k2-2]<<endl;
			  seed1[t1]=Point2f(k1,k2-2);
			  t1=t1+1;//t1为所有种子点数目
				  
			  if ( selfsim[k2-2]>thre_H)
			  {			 			  
				  if(nearsim[k2-2]>thre_L)
				  {
					  t2=t2+1;
					  seed.push_back(k2);

				  }
				  else

				   {   		
					  cv::circle(src, contours[k1][k2-2], 2, cv::Scalar(0, 0, 255));//在图像中画出特征点，2是圆的半径 
					  
				   }
			  }
			 else
			 {        
				     // cv::circle(src, contours[k1][k2-2], 2, cv::Scalar(0, 0, 255));
			 }
			   
		    }
	  
	Num[k1]=t2;
	t2=0;    
    for(int i=0;i<Num[k1];i++)
	 {
		   if (Num[k1]<mini_SegLength)      
		    {
		        cv::circle(src, contours[k1][seed[i]], 2, cv::Scalar(0, 0, 255)); 
		    }
	 }

	 }
	 
 }
 
 imshow("噪声点图像",src);
 waitKey(0);
 return 0;
 
}


 