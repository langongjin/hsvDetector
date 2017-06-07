#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/time.h>

using namespace std;
using namespace cv;

Mat img, imgF, imgHSV, imgRed, imgRed1, imgRed2, imgGreen, imgGreen1, imgGreen2, imgBlue, imgYellow;

void colorDetector(Mat img)
{
    medianBlur(img, imgF, 5); // median filting image with 5*5 size

    int rLowH1=0,rHighH1=10,rLowH2=170,rHighH2=180,rLowS=160,rHighS=255,rLowV=120,rHighV=255; //red
    int gLowH1=35,gHighH1=40,gLowH2=41,gHighH2=59,gLowS1=140,gLowS2=69,gHighS=255,gLowV=104,gHighV=255; //green
    int bLowH=99,bHighH=121,bLowS=120,bHighS=255,bLowV=57,bHighV=211; //blue
    int yLowH=26,yHighH=32,yLowS=130,yHighS=255,yLowV=150,yHighV=255; // yellow

    cvtColor(imgF, imgHSV, COLOR_BGR2HSV); //convert the RGB image to HSV, opencv is BGR
    //H = imgHSV(:,:,1);
    //S = imgHSV(:,:,2);
    //V = imgHSV(:,:,3);

    inRange(imgHSV, Scalar(rLowH1,rLowS, rLowV),Scalar(rHighH1,rHighS, rHighV),imgRed1); //detecting red
    inRange(imgHSV, Scalar(rLowH2,rLowS, rLowV),Scalar(rHighH2,rHighS, rHighV),imgRed2);
    inRange(imgHSV, Scalar(gLowH1,gLowS1, gLowV),Scalar(gHighH1,gHighS, gHighV),imgGreen1); //detecting green
    inRange(imgHSV, Scalar(gLowH2,gLowS2, gLowV),Scalar(gHighH2,gHighS, gHighV),imgGreen2);
    inRange(imgHSV, Scalar(bLowH,bLowS, bLowV),Scalar(bHighH,bHighS, bHighV),imgBlue); //detecting Blue
    inRange(imgHSV, Scalar(yLowH,yLowS, yLowV),Scalar(yHighH,yHighS, yHighV),imgYellow); // detecting yellow

    add(imgRed1, imgRed2, imgRed);
    add(imgGreen1, imgGreen2, imgGreen);
    //threshold(imgBlue,imgBlue,1,1,THRESH_BINARY);

    vector<vector<Point> > contoursBlue, contoursRed,contoursGreen,contoursYellow; //define the 2D point vector to save the coordinate (x,y) of contours
    //CV_RETR_EXTERNAL:similar with nohole, just detect the outside contours, ignore the hole inside.CV_RETR_CCOMP:all Contours
    //CV_CHAIN_APPROX_NONE:save all points of contours, CV_CHAIN_APPROX_SIMPLE:save the vertical and horizontal points.
    //more details about finContours and drawContours:http://blog.csdn.net/maomao1011120756/article/details/49794997
    findContours(imgBlue,contoursBlue,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
    findContours(imgGreen,contoursGreen,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
    findContours(imgRed,contoursRed,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
    findContours(imgYellow,contoursYellow,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);

    int jB = 0, jG = 0, jR = 0, jY = 0, imgBlueArea [50], imgGreenArea [50], imgRedArea [50], imgYellowArea [50];
    int minCenterCoorBlueY = 600, maxCenterCoorBlueY = 0, minCenterCoorBlueX = 1600, maxCenterCoorBlueX = 0; // the max of (x,y) is (800,600)
    int minCenterCoorGreenY = 600, maxCenterCoorGreenY = 0, minCenterCoorGreenX = 1600, maxCenterCoorGreenX = 0;
    int minCenterCoorRedY = 600, maxCenterCoorRedY = 0, minCenterCoorRedX = 1600, maxCenterCoorRedX = 0;
    int minCenterCoorYellowY = 600, maxCenterCoorYellowY = 0, minCenterCoorYellowX = 1600, maxCenterCoorYellowX = 0;
    //int minCenterCoorX = 0, minCenterCoorY = 0, maxCenterCoorX = 0, maxCenterCoorY = 0;
    int heightBlueMax = 0, heightGreenMax = 0, heightRedMax = 0, heightYellowMax = 0;
    Rect rectCoor[100], rectCoorBlue[10], rectCoorGreen[10],rectCoorRed[10],rectCoorYellow[10];
    for (int i = 0; i < contoursBlue.size(); i++)  //calculate the area, center of block and robot, boundaries/rectangles of block and robot
    {
        double imgBlueAreaBuf = contourArea(contoursBlue[i]); //contours: the points of contours
        //Moments imgBlueAreaBuf = moments(contoursBlue[i]); //imgBlueAreaBuf.m00 is the same area with contourArea
        if (imgBlueAreaBuf > 60){
            imgBlueArea[jB] = imgBlueAreaBuf; //convert double to int

            //drawContours(imgF,contoursBlue,i,Scalar(0,0,255),1); //BGR, draw i-th contour
            //cout << "Blue Area -" << jB << "- is: " << imgBlueArea[jB] << endl;
            rectCoorBlue[jB] = boundingRect(contoursBlue[i]);
            rectCoor[jB] = rectCoorBlue[jB];
            //rectangle(imgF, rectCoor, Scalar(0,255,0),1);

            //cout << "the top left coordination is: " << rectCoor.tl() << endl;
            Point centerCoorBlue [50];
            centerCoorBlue[jB].x = (rectCoorBlue[jB].tl().x + rectCoorBlue[jB].br().x)/2;
            centerCoorBlue[jB].y = (rectCoorBlue[jB].tl().y + rectCoorBlue[jB].br().y)/2;
            circle(imgF,centerCoorBlue[jB],2,Scalar(0,255,0),2);
            char textBlue[64];
            snprintf(textBlue, sizeof(textBlue), "%d", jB);
            putText(imgF, textBlue, rectCoor[jB].tl(),FONT_HERSHEY_DUPLEX,0.4,Scalar(0,255,0),1);
            //cout << "---center point blue -" << jB << "- is---" << centerCoorBlue[jB] << endl;
            //contoursBlue.erase(contoursBlue.begin() + i);
            //minCenterCoorBlueY = min(rectCoorBlue[jB].tl().y, minCenterCoorBlueY);
            //maxCenterCoorBlueY = max(rectCoorBlue[jB].br().y, maxCenterCoorBlueY);
            //minCenterCoorBlueX = min(rectCoorBlue[jB].tl().x, minCenterCoorBlueX);
            //maxCenterCoorBlueX = max(rectCoorBlue[jB].br().x, maxCenterCoorBlueX);
            //heightBlueMax = max(rectCoorBlue[jB].height, heightBlueMax);
            jB++;
        }
        //contoursBlue[i].erase(contoursBlue[i].begin(),contoursBlue[i].end());
    }
    //----------Green-----------
    for (int i = 0; i < contoursGreen.size(); i++)  //calculate the area, center of block and robot, boundaries/rectangles of block and robot
    {
        double imgGreenAreaBuf = contourArea(contoursGreen[i]); //contours: the points of contours
        Point centerCoorGreen[50];
        if (imgGreenAreaBuf > 60){
            imgGreenArea[jG] = imgGreenAreaBuf; //convert double to int

            rectCoorGreen[jG] = boundingRect(contoursGreen[i]);
            rectCoor[jB+jG] = rectCoorGreen[jG];

            centerCoorGreen[jG].x = (rectCoorGreen[jG].tl().x + rectCoorGreen[jG].br().x)/2;
            centerCoorGreen[jG].y = (rectCoorGreen[jG].tl().y + rectCoorGreen[jG].br().y)/2;
            circle(imgF,centerCoorGreen[jG],2,Scalar(0,255,0),2);
            char textGreen[64];
            snprintf(textGreen, sizeof(textGreen), "%d", jB+jG);
            putText(imgF, textGreen, rectCoor[jB+jG].tl(),FONT_HERSHEY_DUPLEX,0.4,Scalar(0,255,0),1);
            //cout << "---center point green -" << jG << "- is---" << centerCoorGreen[jG] << endl;
            //minCenterCoorGreenY = min(rectCoorGreen[jG].tl().y, minCenterCoorGreenY);
            //maxCenterCoorGreenY = max(rectCoorGreen[jG].br().y, maxCenterCoorGreenY);
            //minCenterCoorGreenX = min(rectCoorGreen[jG].tl().x, minCenterCoorGreenX);
            //maxCenterCoorGreenX = max(rectCoorGreen[jG].br().x, maxCenterCoorGreenX);
            //heightBlueMax = max(rectCoorGreen[jB].height, heightGreenMax);
            jG++;
        }
    }
    //---------Red----------
    for (int i = 0; i < contoursRed.size(); i++)  //calculate the area, center of block and robot, boundaries/rectangles of block and robot
    {
        double imgRedAreaBuf = contourArea(contoursRed[i]); //contours: the points of contours

        if (imgRedAreaBuf > 80){
            imgRedArea[jR] = imgRedAreaBuf; //convert double to int

            //drawContours(imgF,contoursRed,i,Scalar(0,0,255),1); //BGR, draw i-th contour
            //cout << "Red Area " << jR << " is: " << imgRedArea[jR] << endl;
            rectCoorRed[jR] = boundingRect(contoursRed[i]);
            rectCoor[jB+jG+jR] = rectCoorRed[jR];
            //rectangle(imgF, rectCoor, Scalar(0,255,0),1);
            char textRed[64];
            snprintf(textRed, sizeof(textRed), "R%d:%d", jR, imgRedArea[jR]);
            putText(imgF, textRed, rectCoorRed[jR].tl(),FONT_HERSHEY_DUPLEX,0.4,Scalar(0,255,0),1);
            //cout << "the top left coordination is: " << rectCoor.tl() << endl;
            Point centerCoorRed[50];
            centerCoorRed[jR].x = (rectCoorRed[jR].tl().x + rectCoorRed[jR].br().x)/2;
            centerCoorRed[jR].y = (rectCoorRed[jR].tl().y + rectCoorRed[jR].br().y)/2;
            circle(imgF,centerCoorRed[jR],2,Scalar(0,255,0),2);
            //cout << "---center point red -" << jR << "- is---" << centerCoorRed[jR] << endl;
            //minCenterCoorRedY = min(rectCoorRed[jR].tl().y, minCenterCoorRedY);
            //maxCenterCoorRedY = max(rectCoorRed[jR].br().y, maxCenterCoorRedY);
            //minCenterCoorRedX = min(rectCoorRed[jR].tl().x, minCenterCoorRedX);
            //maxCenterCoorRedX = max(rectCoorRed[jR].br().x, maxCenterCoorRedX);
            jR++;
        }
    }
    //----------Yellow----------
    for (int i = 0; i < contoursYellow.size(); i++)  //calculate the area, center of block and robot, boundaries/rectangles of block and robot
    {
        double imgYellowAreaBuf = contourArea(contoursYellow[i]); //contours: the points of contours

        if (imgYellowAreaBuf > 100){
            imgYellowArea[jY] = imgYellowAreaBuf; //convert double to int

            //drawContours(imgF,contoursYellow,i,Scalar(0,0,255),1); //BGR, draw i-th contour
            //cout << "Yellow Area " << jY << " is: " << imgYellowArea[jY] << endl;
            rectCoorYellow[jY] = boundingRect(contoursYellow[i]);
            rectCoor[jB+jG+jR+jY] = rectCoorYellow[jY];
            //rectangle(imgF, rectCoor, Scalar(0,255,0),1);
            char textYellow[64];
            snprintf(textYellow, sizeof(textYellow), "Y%d:%d", jY, imgYellowArea[jY]);
            //putText(imgF, textYellow, rectCoorYellow.tl(),FONT_HERSHEY_DUPLEX,0.4,Scalar(0,255,0),1);
            //cout << "the top left coordination is: " << rectCoor.tl() << endl;
            Point centerCoorYellow[50];
            centerCoorYellow[jY].x = (rectCoorYellow[jY].tl().x + rectCoorYellow[jY].br().x)/2;
            centerCoorYellow[jY].y = (rectCoorYellow[jY].tl().y + rectCoorYellow[jY].br().y)/2;
            circle(imgF,centerCoorYellow[jY],2,Scalar(0,255,0),2);
            //cout << "---center point yellow -" << jY << "- is---" << centerCoorYellow[jY] << endl;
            jY++;
            //minCenterCoorYellowY = min(rectCoorYellow[jY].tl().y, minCenterCoorYellowY);
            //maxCenterCoorYellowY = max(rectCoorYellow[jY].br().y, maxCenterCoorYellowY);
            //minCenterCoorYellowX = min(rectCoorYellow[jY].tl().x, minCenterCoorYellowX);
            //maxCenterCoorYellowX = max(rectCoorYellow[jY].br().x, maxCenterCoorYellowX);
        }
    }
    //---------define include distance-----------
    int num = jB+jG+jR+jY;
    cout << "number : " << num << endl;
    //int thresholdBoxNum = num < 4 ? 1 : (num < 6 ? num * 0.6 : num * 0.4); // define
    //cout << "thresholdBoxNum : " << thresholdBoxNum << endl;
    int distanceBox[num][num], distanceBoxSum[num], numBox[num], minDistanceBox[num], min2DistanceBox[num],rectBoxHeight = 0, rectBoxHeightMax = 0;

    for (int i = 0; i < num; i++) //calculating the suitable(medium) value of height
    {
        if (rectCoor[i].height > rectBoxHeightMax)
        {
            rectBoxHeight = rectBoxHeightMax; // set this value as the height of box
            rectBoxHeightMax = rectCoor[i].height;
        }
        else if (rectCoor[i].height > rectBoxHeight)
            rectBoxHeight = rectCoor[i].height;
        //cout << "rectBoxHeight : " << rectBoxHeight << " --- rectBoxHeightMax : " << rectBoxHeightMax << endl;
    }

    for (int j = 0; j < num; j++) //calculating the value of minimum and the second minimum distance for each box
    {
        minDistanceBox[j] = 1600;
        min2DistanceBox[j] = 1600;
        for (int x = 0; x < num; x++)
        {
            if (j != x)
            {
                distanceBox[j][x] = min(abs(rectCoor[j].tl().x - rectCoor[x].br().x),abs(rectCoor[j].br().x - rectCoor[x].tl().x));
                //distanceBox[j][x] = abs(rectCoor[j].x + rectCoor[j].width - rectCoor[x].x - rectCoor[x].width);
                cout << "distanceBox [" << j << ", " << x << "] : " << distanceBox[j][x] << endl;
                if (distanceBox[j][x] < minDistanceBox[j])
                {
                    min2DistanceBox[j] = minDistanceBox[j]; //the second minimum distance
                    cout << "min2DistanceBox " << j << " :" << min2DistanceBox[j] << endl;
                    minDistanceBox[j] = distanceBox[j][x]; //the minimun distance
                    cout << "minDistanceBox " << j << " :" << minDistanceBox[j] << endl;
                }
                else if (distanceBox[j][x] < min2DistanceBox[j])
                {
                    min2DistanceBox[j] = distanceBox[j][x];
                    cout << "min2DistanceBox " << j << " :" << min2DistanceBox[j] << endl;
                    cout << "minDistanceBox " << j << " :" << minDistanceBox[j] << endl;
                }
            }
        }
        distanceBoxSum[j] = minDistanceBox[j] + min2DistanceBox[j];
        cout << "-------------------------distanceBoxSum : " << distanceBoxSum[j] << endl;
    }

    for (int i =0; i < num; i++) //sequence from minimum distance to maximum distance
    {
        numBox[i] = 0;
        for (int j=0; j < num; j++)
        {
            if (i != j) // get the Box[i] sequence
            {
                if (distanceBoxSum[i] > distanceBoxSum[j])
                    numBox[i] = numBox[i] +1; //save the number
                if (distanceBoxSum[i] == distanceBoxSum[j])
                {
                    if (minDistanceBox[i] >= minDistanceBox[j]) //always have the same distance between two points each other
                        numBox[i] = numBox[i] +1;
                }
            }
        }
        cout << "numBox " << i << " :" << numBox[i] << endl;
    }
    //-------------difine the robot------------
    int lastnum = num, robNum, minRectCoorX[num], minRectCoorY[num], maxRectCoorX[num], maxRectCoorY[num];
    for (robNum = 0; lastnum >= 2 && robNum < num; robNum++)
    {
        int minNumBox=100;
        cout << "-------------------------------------------------robNum :" << robNum << " robot -------------------------------------" << endl;
        //lastnum --;
        for (int k = 0; k <num; k++) //get the minNumBox between the rest
        {
            cout << "numBox " << k << "---------" << numBox[k] << endl;
            minNumBox = min(numBox[k], minNumBox);
            cout << "minNumBox " << k << ": " << minNumBox << endl;
        }
        cout << "--------fucking------------" << endl;
        for (int i = 0; i < num; i++) //get the coordination of rectangle of robot from boxes
        {
            if (numBox[i] == minNumBox) //find the minimum one between the rest (usually it is 0 when 1 robot)
            {
                lastnum --;
                if (num > 2) //when robot only have 2 boxes at least, just combine the two boxes
                    numBox[i] = 100; //make it not included in the rest
                minRectCoorX[robNum] = rectCoor[i].tl().x;
                minRectCoorY[robNum] = rectCoor[i].tl().y;
                maxRectCoorX[robNum] = rectCoor[i].br().x;
                maxRectCoorY[robNum] = rectCoor[i].br().y;
                cout << "------numBox " << i << ": " << numBox[i] << endl;
                int bufnum = 0, jBox[50] = {0};
                for (int j = 0; j < num; j++) //calculating the coordination of rectangle incluing boxes belong to the distance area
                {
                    cout << "---------------------------mark------------------ j = " << j << endl;
                    cout << "numBox " << j << " : " << numBox[j] << endl;
                    cout << "distanceBox " << i << "," << j << " : " << distanceBox[i][j] << endl;
                    cout << "rectBoxHeight : " << rectBoxHeight << endl;
                    //-------------the first threshold condition-------------------
                    if (j != i && numBox[j] != 100 && distanceBox[i][j] < 4.3 * rectBoxHeight) //3.4, 3.5, 4.5, 4.3 justify if the box belong to the same robot by distance of boxeswith the center box
                    {

                        jBox[bufnum] = j;
                        cout << "jBox " << bufnum << " :" << jBox[bufnum] << "---------------" << endl;
//                        cout << "minRectCoorX " << robNum << ": " << minRectCoorX[robNum] << endl;
//                        cout << "minRectCoorY " << robNum << ": " << minRectCoorY[robNum] << endl;
//                        cout << "maxRectCoorX " << robNum << ": " << maxRectCoorX[robNum] << endl;
//                        cout << "maxRectCoorY " << robNum << ": " << maxRectCoorY[robNum] << endl;
//
//                        minRectCoorX[robNum] = min(rectCoor[j].tl().x, minRectCoorX[robNum]);
//                        minRectCoorY[robNum] = min(rectCoor[j].tl().y, minRectCoorY[robNum]);
//                        maxRectCoorX[robNum] = max(rectCoor[j].br().x, maxRectCoorX[robNum]);
//                        maxRectCoorY[robNum] = max(rectCoor[j].br().y, maxRectCoorY[robNum]);
//                        cout << "--------" << endl;
//                        cout << "minRectCoorX " << robNum << ": " << minRectCoorX[robNum] << endl;
//                        cout << "minRectCoorY " << robNum << ": " << minRectCoorY[robNum] << endl;
//                        cout << "maxRectCoorX " << robNum << ": " << maxRectCoorX[robNum] << endl;
//                        cout << "maxRectCoorY " << robNum << ": " << maxRectCoorY[robNum] << endl;
                        //numBox[j] = 100; //set a constant not zero and more than all of the numBox
                        lastnum --;
                        cout << "lastnum ----------------" << lastnum << endl;
                        bufnum ++; //the number of boxes that match the threshold of (distanceBox[i][j] < 3.4 * rectBoxHeight)
                        cout << "bufnum ----------------" << bufnum << endl;
                    }
                    //----calculating the max distance between boxes after the first threshold condition, preparing for next--------
                    if (j == num - 1 && bufnum >= 1) //bufnum >= 1 (it have two candidate at least)
                    {
                        cout << "bufnum : " << bufnum << endl;
                        cout << "------------------5----------jBox[0] = " << jBox[0] << endl;
                        int maxBoxDisOut[num], max_in_out[num][num],maxBoxDisOutNum[num];
                        //bufnum >1 guarantte the robot have center box and two other box at least, or not go to compare center box and another one box
                        for (int buf = 0; buf < bufnum; buf++) //calculating the max distance between boxes in jBox[bufnum]
                        {
                            cout << "------------------55----------jBox[0] = " << jBox[0] << endl;
                            cout << "--------------------------1------------buf : " << buf << endl;
                            maxBoxDisOut[jBox[buf]] = 0;
                            cout << "-------------- bufnum = " << bufnum << " ------------" << endl;
                            cout << "------------------555----------jBox[0] = " << jBox[0] << endl;
                            int rectCoor_tl_br, rectCoor_br_tl;
                            if (bufnum == 1) // one other box and one center box
                            {
                                cout << "-----------------i = " << i << endl;
                                rectCoor_tl_br = abs(rectCoor[i].tl().x - rectCoor[jBox[0]].br().x); //calculating the inside or outside distance between the same boxes
                                cout << "rectCoor_tl_br : " << rectCoor_tl_br << "---------------6------" << endl;
                                rectCoor_br_tl = abs(rectCoor[i].br().x - rectCoor[jBox[0]].tl().x); //calculating the inside or outside distance between the same boxes
                                cout << "rectCoor_br_tl : " << rectCoor_br_tl << "---------------7------" << endl;
                                maxBoxDisOut[jBox[0]] = min(rectCoor_tl_br,rectCoor_br_tl); //max, min
                                cout << "maxBoxDisOut " << jBox[0] << " = " << maxBoxDisOut[jBox[0]] << endl;
                            }
                            else
                            {
                                for (int buff = 0; buff < bufnum; buff++)
                                {
                                    cout << "-----------------------------------------------3---------------buff : " << buff << endl;
                                    rectCoor_tl_br = abs(rectCoor[jBox[buf]].tl().x - rectCoor[jBox[buff]].br().x); //calculating the inside or outside distance between the same boxes
                                    cout << "rectCoor_tl_br : " << rectCoor_tl_br << "----" << jBox[buf] << "----" << jBox[buff] << "---------------6------" << endl;
                                    rectCoor_br_tl = abs(rectCoor[jBox[buf]].br().x - rectCoor[jBox[buff]].tl().x); //calculating the inside or outside distance between the same boxes
                                    cout << "rectCoor_br_tl : " << rectCoor_br_tl << "---------------7------" << endl;
                                    max_in_out[jBox[buf]][jBox[buff]] = min(rectCoor_tl_br,rectCoor_br_tl); //max,min
                                    //cout << "-------max_in_out[jBox[buf]][jBox[buff]] : " << max_in_out[jBox[buf]][jBox[buff]] << endl;
                                    //distanceBox[j][x] = abs(rectCoor[j].x + rectCoor[j].width - rectCoor[x].x - rectCoor[x].width);
                                    if (max_in_out[jBox[buf]][jBox[buff]] > maxBoxDisOut[jBox[buf]])
                                    {
                                        maxBoxDisOut[jBox[buf]] = max_in_out[jBox[buf]][jBox[buff]];
                                        cout << "maxBoxDisOut " << jBox[buf] << " : " << maxBoxDisOut[jBox[buf]] << "--------------------8-----" << endl;
                                        maxBoxDisOutNum[buf] = jBox[buff];
                                        cout << "maxBoxDisOutNum " << buf << " : " << maxBoxDisOutNum[buf] << " ----------------9------" << endl;
                                    }
                                }
                            }
                        }
                        //bufnum >1 guarantte the robot have center box and two other box (bufnum=2) at least, or not go to compare center box and another one box
                        if (bufnum >= 2)
                        {
                            int delNum = 0;
                            for (int bufff = 0; bufff < bufnum; bufff++) //compare the max distance (robot size from left to right) of boxes in jBox[bufnum]
                            {
                                cout << "---------------------------4---------- i = " << i<< endl;
                                cout << "bufff : " << bufff << " ------------------------2-------" << endl;
                                cout << "maxBoxDisOut[jBox[bufff]] : jBox[bufff] = " << jBox[bufff] << " ----- " << maxBoxDisOut[jBox[bufff]] << endl;
                                cout << distanceBox[i][jBox[bufff]] << "-----"<<maxBoxDisOutNum[bufff] << " ----- " << distanceBox[i][maxBoxDisOutNum[bufff]]<< endl;
                                if (maxBoxDisOut[jBox[bufff]] < 6.2 * rectBoxHeight) //if > the length of robot, delete far one, get the near one as rectangle
                                {
                                    cout << "------------------5----------maxBoxDisOut[jBox[bufff]]: jBox[bufff] = " << jBox[bufff] << endl;
                                    minRectCoorX[robNum] = min(rectCoor[jBox[bufff]].tl().x, minRectCoorX[robNum]);
                                    minRectCoorY[robNum] = min(rectCoor[jBox[bufff]].tl().y, minRectCoorY[robNum]);
                                    maxRectCoorX[robNum] = max(rectCoor[jBox[bufff]].br().x, maxRectCoorX[robNum]);
                                    maxRectCoorY[robNum] = max(rectCoor[jBox[bufff]].br().y, maxRectCoorY[robNum]);
                                    numBox[jBox[bufff]] = 100; //set a constant not zero and more than all of the numBox
                                    cout << "-----------------------------------------------------mark------------------ bufff = " << bufff << endl;
                                    cout << "lastnum ----------------" << lastnum << endl;
                                    cout << "bufnum ----------------" << bufnum << endl;
                                }
                                else if (distanceBox[i][jBox[bufff]] < distanceBox[i][maxBoxDisOutNum[bufff]]) //always have two boxes match this condition at the same time, choice one of them
                                {
                                    //继续比较较小distanceBox的第二maxBoxDisOut是否满足8.5，如果满足，执行下面，如果不满足，则什么都不做（即两个对应的maxBoxDisOut都不符合）
                                    minRectCoorX[robNum] = min(rectCoor[jBox[bufff]].tl().x, minRectCoorX[robNum]);
                                    minRectCoorY[robNum] = min(rectCoor[jBox[bufff]].tl().y, minRectCoorY[robNum]);
                                    maxRectCoorX[robNum] = max(rectCoor[jBox[bufff]].br().x, maxRectCoorX[robNum]);
                                    maxRectCoorY[robNum] = max(rectCoor[jBox[bufff]].br().y, maxRectCoorY[robNum]);
                                    numBox[jBox[bufff]] = 100; //set a constant not zero and more than all of the numBox
                                    //lastnum ++; //plus for the cancelled more one
                                    //bufnum --;
                                    cout << "-----------------------------------------------------mark1------------------ bufff = " << bufff << endl;
                                    cout << "lastnum ----------------" << lastnum << endl;
                                    cout << "bufnum ----------------" << bufnum << endl;
                                }
                                else
                                {

                                    minRectCoorX[robNum] = min(rectCoor[maxBoxDisOutNum[bufff]].tl().x, minRectCoorX[robNum]);
                                    minRectCoorY[robNum] = min(rectCoor[maxBoxDisOutNum[bufff]].tl().y, minRectCoorY[robNum]);
                                    maxRectCoorX[robNum] = max(rectCoor[maxBoxDisOutNum[bufff]].br().x, maxRectCoorX[robNum]);
                                    maxRectCoorY[robNum] = max(rectCoor[maxBoxDisOutNum[bufff]].br().y, maxRectCoorY[robNum]);
                                    numBox[maxBoxDisOutNum[bufff]] = 100;
                                    delNum ++;
                                    cout << "-----------------------------------------------------mark2------------------ bufff = " << bufff << endl;
                                    cout << "delNnum ----------------" << delNum << endl;
                                }
                            }
                            lastnum = lastnum + delNum; //plus for the cancelled more one
                            bufnum = bufnum - delNum;
                            cout << "lastnum ----------------" << lastnum << endl;
                            cout << "bufnum ----------------" << bufnum << endl;
                        }
                        else //compare center box and another one box, when bufnum = 1
                        {
                            cout << "------------------11----------jBox[0] = " << jBox[0] << endl;
                            cout << "maxBoxDisOut " << jBox[0] << " : " << maxBoxDisOut[jBox[0]] << endl;
                            if (maxBoxDisOut[jBox[0]] < 6.2 * rectBoxHeight) //the length of robot 9.4
                            {
                                cout << "----------------------------jBox[0] = " << jBox[0] << endl;
                                minRectCoorX[robNum] = min(rectCoor[jBox[0]].tl().x, minRectCoorX[robNum]);
                                minRectCoorY[robNum] = min(rectCoor[jBox[0]].tl().y, minRectCoorY[robNum]);
                                maxRectCoorX[robNum] = max(rectCoor[jBox[0]].br().x, maxRectCoorX[robNum]);
                                maxRectCoorY[robNum] = max(rectCoor[jBox[0]].br().y, maxRectCoorY[robNum]);
                                numBox[jBox[0]] = 100; //set a constant not zero and more than all of the numBox
                            }
                            else //just one center to rest
                            {
                                robNum --;
                                cout << "-------------------------mark robNum-----------------" << endl;
                            }
                        }
                    }
                }
            }
        }
    }

    for (int i = 0; i < robNum; i++)
    {
        cout << "----------------------------- draw robot rectangle : " << i << endl;
        rectangle(imgF, Point(minRectCoorX[i],minRectCoorY[i]),Point(maxRectCoorX[i],maxRectCoorY[i]),Scalar(0,255,0),1);

        int robCenterCoorX = (minRectCoorX[i] + maxRectCoorX[i])/2;
        int robCenterCoorY = (minRectCoorY[i] + maxRectCoorY[i])/2;
        circle(imgF,Point(robCenterCoorX,robCenterCoorY),3,Scalar(0,255,0),4);
        char textRobCenterCoor[64];
        snprintf(textRobCenterCoor, sizeof(textRobCenterCoor),"(%d,%d)",robCenterCoorX,robCenterCoorY);
        putText(imgF, textRobCenterCoor, Point(robCenterCoorX + 10,robCenterCoorY+3),FONT_HERSHEY_DUPLEX,0.4,Scalar(0,255,0),1);
        cout << robCenterCoorX + 10 << "--" << robCenterCoorY+3 << endl;
        line(imgF, Point(minRectCoorX[i],minRectCoorY[i]), Point(maxRectCoorX[i],maxRectCoorY[i]),Scalar(0,255,0),1);
        line(imgF, Point(minRectCoorX[i],maxRectCoorY[i]), Point(maxRectCoorX[i],minRectCoorY[i]),Scalar(0,255,0),1);
    }
    //namedWindow("image", CV_WINDOW_NORMAL); //CV_WINDOW_NORMAL, WINDOW_AUTOSIZE
    imshow("image", imgF);
    //waitKey(0);
}

int main() {
    struct timeval timeStart, timeEnd;
    double timeDiff;

    VideoCapture cap(0); //open the first camera

    if(!cap.isOpened()) //check for successful open, or not
        cerr << "Can not open a camera or file." << endl;

    int rate = cap.get(CV_CAP_PROP_FPS); //获取帧率

    cout << "frame of camera : " << rate << endl;

    bool stop = false; //定义一个用来控制读取视频循环结束的变量
    while(!stop)
    {
        cap >> img; //read a frame image and save to the Mat img

        gettimeofday(&timeStart,NULL);
        colorDetector(img);
        gettimeofday(&timeEnd,NULL);
        timeDiff = 1000*(timeEnd.tv_sec - timeStart.tv_sec) + (timeEnd.tv_usec - timeStart.tv_usec)/1000; //tv_sec: value of second, tv_usec: value of microsecond
        cout << "Time for one frame : " << timeDiff << " ms" << endl;

        int c = waitKey(10); //waiting for 5 milliseconds to detect the pression, if waitKey(0) meaning always waiting until any pression
        if((char) c == 27) // press escape to exit the imshow
        {
            stop = true;
        }
    }
    //waitKey(); //注意：imshow之后必须加waitKey，否则无法显示图像
    return 0;
}