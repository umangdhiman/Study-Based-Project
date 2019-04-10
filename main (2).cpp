#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/nonfree/features2d.hpp> //
//#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <dirent.h>
#include <ctime>
using namespace cv;
using namespace std;
std::vector<DMatch> filter_distance(Mat descriptors,std::vector< DMatch > matches);


int main(int argc, const char *argv[]){
    int ch;
    if(argc != 3){
        cout << "usage:match <image1> <image2>\n" ;
        exit(-1);
    }
    string image1_name=string(argv[1]), image2_name = string(argv[2]);

    Mat image1 = imread(image1_name, CV_LOAD_IMAGE_GRAYSCALE );
    Mat image2 = imread(image2_name, CV_LOAD_IMAGE_GRAYSCALE );

    if( !image1.data || !image2.data ){
        std::cout<< " --(!) Error reading images " << std::endl; return -1;
    }
    std::vector<KeyPoint> kp,kp2;

    int nfeatures=10000;
    float scaleFactor=1.414f;
    int nlevels=5;
    int edgeThreshold=15; // Changed default (31);
    int firstLevel=0;
    int WTA_K=2;
    int scoreType=ORB::HARRIS_SCORE;
    int patchSize=31;
    int fastThreshold=20;

    Ptr<ORB> detector = ORB::create(
    nfeatures,
    scaleFactor,
    nlevels,
    edgeThreshold,
    firstLevel,
    WTA_K,
    scoreType,
    patchSize,
    fastThreshold );
    Ptr<ORB> extractor=ORB::create(nfeatures,scaleFactor,nlevels,edgeThreshold,firstLevel,WTA_K,scoreType,patchSize,fastThreshold);

    detector->detect(image1, kp);
    detector->detect(image2,kp2);
    // TODO default is 500 keypoints..but we can change

    cout << "# keypoints of image1 :" << kp.size() << endl;
    cout << "# keypoints of image2 :" << kp2.size() << endl;

    Mat descriptors1,descriptors2;
    extractor->compute(image1,kp,descriptors1);
    extractor->compute(image2,kp2,descriptors2);
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    while(1){
        cout<<"Enter 1 for Bruteforce match\nEnter 2 for match by filtering distance\nEnter 3 for Flann-Based matching\nEnter 4 for ratio based matching\nEnter 5 for improved ORB\nEnter Any other key to exit\n";
        cin>>ch;

        switch(ch){
        case 1:{
            std::vector< DMatch > matches, matches2;
            clock_t begin = clock();
            matcher->match( descriptors1, descriptors2, matches);
            matcher->match( descriptors2, descriptors1, matches2);
  //-- Draw matches
            clock_t end = clock();
            double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
            cout << "Time Costs : " << elapsed_secs << endl;
            Mat img_matches, img_matches2;
            drawMatches( image1, kp, image2, kp2,matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
            //drawMatches( image1, kp, image2, kp2,matches2, img_matches2, Scalar::all(-1), Scalar::all(-1),
              // vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
            cout << "Matches1-2:" << matches.size() << endl;
            cout << "Matches2-1:" << matches2.size() <<endl;
            namedWindow("Matches result 1-2",WINDOW_KEEPRATIO);
            imshow("Matches result 1-2",img_matches);
            //imshow("Matches result 2-1", img_matches2);
            waitKey(0);
            destroyAllWindows();
            break;
        }
        case 2:
        {
            int t;
            cin>>t;
            vector< vector<DMatch> > matchesx, matchesz;
            clock_t begin = clock();
            matcher->knnMatch( descriptors1, descriptors2, matchesx,t );
            matcher->knnMatch( descriptors2, descriptors1, matchesz, t);
            cout << "Matches 1 to 2 ==" << matchesx.size() << ", "<< matchesx[0].size() << endl;
            cout << "Matches 2 to 1 ==" << matchesz.size() << ", "<< matchesz[0].size() << endl;
            clock_t end = clock();
            double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
            cout << "Time Costs : " << elapsed_secs << endl;
         //   Mat img_matches,img_matches2;

           /* drawMatches(image1,kp,image2,kp2,matchesx[0],img_matches);
            imshow("Matches",img_matches);
            drawMatches(image1,kp,image2,kp2,matchesz[0],img_matches2);
            imshow("Matches2",img_matches2);*/
            waitKey(0);
          //  destroyAllWindows();
            break;}
        case 3:
        {
            std::vector< DMatch > matches, matches2;
            clock_t begin = clock();
            matcher->match( descriptors1, descriptors2, matches );
            matcher->match( descriptors2, descriptors1, matches2 );

            std::vector< DMatch > good_matches1, good_matches2, better_matches;
            good_matches1 = filter_distance(descriptors1, matches);
            good_matches2 = filter_distance(descriptors2, matches2);

            for(int i=0; i<good_matches1.size(); i++){
                DMatch temp1 = good_matches1[i];
                for(int j=0; j<good_matches2.size(); j++){
                    DMatch temp2 = good_matches2[j];
                    if(temp1.queryIdx == temp2.trainIdx && temp2.queryIdx == temp1.trainIdx) {
                        better_matches.push_back(temp1);
                        break;
                    }
                }
            }
            clock_t end = clock();
            double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
            cout << "Time Costs : " << elapsed_secs << endl;
            //-- Draw only "good" matches
            Mat img_matches;
            drawMatches( image1, kp, image2, kp2,
               better_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
            //-- Show detected matches
            namedWindow("Good matches",WINDOW_KEEPRATIO);
            imshow( "Good Matches", img_matches );
            //for( int i = 0; i < (int)better_matches.size(); i++ )  {
              //  printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, better_matches[i].queryIdx, better_matches[i].trainIdx );
            //}
            cout << "Matches:" << better_matches.size() <<endl;

            waitKey(0);
            destroyAllWindows();
            break;
        }
        case 4:
        {
            vector< vector<DMatch> > matchesx, matchesz;
            double ratio;
            cout<<"Enter ratio\n";
            cin>>ratio;
            clock_t begin = clock();
            matcher->knnMatch( descriptors1, descriptors2, matchesx, 2 );
            matcher->knnMatch( descriptors2, descriptors1, matchesz, 2 );
            cout << "Matches1-2:" << matchesx.size() << endl;
            cout << "Matches2-1:" << matchesz.size() << endl;
// ratio test proposed by David Lowe paper = 0.8
            std::vector<DMatch> good_matches1, good_matches2;
    // Yes , the code here is redundant, itŝs easy to reconstruct it ....
            for(int i=0; i < matchesx.size(); i++){
                if(matchesx[i][0].distance < ratio * matchesx[i][1].distance)
                    good_matches1.push_back(matchesx[i][0]);
            }
            for(int i=0; i < matchesz.size(); i++){
                if(matchesz[i][0].distance < ratio * matchesz[i][1].distance)
                    good_matches2.push_back(matchesz[i][0]);
            }
            cout << "Good matches1:" << good_matches1.size() << endl;
            cout << "Good matches2:" << good_matches2.size() << endl;
    // Symmetric Test
            std::vector<DMatch> better_matches;
            for(int i=0; i<good_matches1.size(); i++){
                for(int j=0; j<good_matches2.size(); j++){
                    if(good_matches1[i].queryIdx == good_matches2[j].trainIdx && good_matches2[j].queryIdx == good_matches1[i].trainIdx){
                        better_matches.push_back(DMatch(good_matches1[i].queryIdx, good_matches1[i].trainIdx, good_matches1[i].distance));
                        break;
                    }
                }
            }
            cout << "Better matches:" << better_matches.size() << endl;
            clock_t end = clock();
            double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
            cout << "Time Costs : " << elapsed_secs << endl;
    // show it on an image
            Mat output;
            drawMatches(image1, kp, image2, kp2, better_matches, output, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
            namedWindow("Matches result",WINDOW_KEEPRATIO);
            imshow("Matches result",output);
            waitKey(0);
            destroyAllWindows();
            break;
        }
        case 5:
        {
            vector<Mat> snapshots;
            vector<DMatch> matches;
            clock_t begin = clock();

            BFMatcher matcher(NORM_HAMMING,true);
            matcher.match(descriptors1,descriptors2,matches);
            int scene_changed=0;
            vector<DMatch> inliers;
            if(matches.size()>100){
                vector<Point2f> srcpoints,dstpoints;
                for(size_t i=0;i<matches.size();i++){
                    srcpoints.push_back(kp[matches[i].queryIdx].pt);
                    dstpoints.push_back(kp2[matches[i].trainIdx].pt);
                }
                vector<uchar> status;
                Mat H=findHomography(srcpoints,dstpoints,status,RANSAC);
                for(size_t i=0;i<matches.size();i++){
                    if(status[i]) inliers.push_back(matches[i]);
                }
                if(inliers.size()<80) scene_changed=1;
            }
            else scene_changed=1;
            if(scene_changed){
                image2.copyTo(image1);
                Ptr<Mat> tmp(new Mat());
                image2.copyTo(*tmp);
                snapshots.push_back(*tmp);
            }
            clock_t end = clock();
            double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
            cout << "Time Costs : " << elapsed_secs << endl;
            cout << "matches:" <<inliers.size() << endl;

            Mat matchImage;
            drawMatches(image1,kp,image2,kp2,inliers,matchImage,Scalar::all(-1),Scalar::all(-1),vector<char>(),DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            namedWindow("Display window",WINDOW_KEEPRATIO);
            imshow("Display window",matchImage);

            waitKey(0);
            destroyAllWindows();
            break;
        }
        default:
            exit(0);
        }
    }
    return 0;
}
std::vector<DMatch> filter_distance(Mat descriptors,std::vector< DMatch > matches){
  double max_dist = 0; double min_dist = 100;

  //-- Quick calculation of max and min distances between keypoints
  for( int i = 0; i < descriptors.rows; i++ )  {
    double dist = matches[i].distance;
    if( dist < min_dist )
      min_dist = dist;
    if( dist > max_dist )
      max_dist = dist;
  }

  printf("-- Max dist : %f \n", max_dist );
  printf("-- Min dist : %f \n", min_dist );

  //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
  //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
  //-- small)
  //-- PS.- radiusMatch can also be used here.
  std::vector< DMatch > good_matches;

  for( int i = 0; i < descriptors.rows; i++ )
  { if( matches[i].distance <= max(2*min_dist,0.02) )
    { good_matches.push_back( matches[i]); }
  }
  return good_matches;
}
