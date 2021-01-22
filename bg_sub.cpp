/**
 * @file bg_sub.cpp
 * @brief fisheye video sorting based on motion metrics from Background subtraction
 * @author David Moore
 */

#include <iostream>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <string>
#include <iomanip>
#include <mutex>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/features2d.hpp>


using namespace cv;
using namespace std;

const string PATHTOVIDEOS = "../videos";
const double MOTIONTHRESH = 3100;
const double DISTTHRESH = 540000;
mutex logMutex;

bool fileExists(string& fileName) {
    return static_cast<bool>(std::ifstream(fileName));
}

template <typename filename, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
bool writeCsvFile(filename& fileName, T1 column1, T2 column2, T3 column3, T4 column4, T5 column5, T6 column6) {
    std::lock_guard<std::mutex> csvLock(logMutex);
    std::fstream file;
    file.open(fileName, std::ios::out | std::ios::app);
    if (file) {
        file << "\"" << column1 << "\",";
        file << "\"" << column2 << "\",";
        file << "\"" << column3 << "\",";
        file << "\"" << column4 << "\",";
        file << "\"" << column5 << "\",";
        file << "\"" << column6 << "\"";
        file << std::endl;
        return true;
    }
    else {
        return false;
    }
}


int main(int argc, char* argv[])
{

    Ptr<BackgroundSubtractor> pBackSub;
    
    TickMeter processingTimer;              

    string csvFile = "results.csv";        
    if (!fileExists(csvFile))
        writeCsvFile(csvFile, "Filename", "Duration", "Metric 1", "Metric 2", "Time of max", "Saved (0 or 1)");

    // Iterate over video files
    for (const auto& entry : experimental::filesystem::directory_iterator(PATHTOVIDEOS)) 
    {
        processingTimer.start();

        VideoCapture capture(entry.path().string());
        if (!capture.isOpened()) {
            cerr << "Unable to open: " << entry.path().string() << endl;
            return 0;
        }

        cout << entry.path() << endl;

        double videoDuration = capture.get(CAP_PROP_FRAME_COUNT) / capture.get(CAP_PROP_FPS);
        cout << "Duration (S): " << videoDuration << endl;

        double sumChannelZero = 0.0;
        double sumDistribution = 0.0;
        double entropyMax = 0.0;
        double sumMax = 0.0;
        double sumMaxTime = 0.0;

        double minVal, maxVal;
        Point minIdx, maxIdx;

        Mat frame, fgMask, saveFrame, saveMask, dist;
        int framenum = 0;
        int motion_frames_count = 0;

        // Region of interest (assumes all videos are 2880x2880 )
        int xo = 200, yo = 200;
        int width = 2480, height = 2480;

        pBackSub = createBackgroundSubtractorMOG2(30, 600, false);

        // Setup SimpleBlobDetector parameters.
        SimpleBlobDetector::Params params;

        // Change thresholds
        params.filterByArea = true;
        params.minArea = 100;
        params.maxArea = 10000;

        params.filterByCircularity = false;
        params.filterByConvexity = false;
        params.filterByInertia = false;

        Ptr<SimpleBlobDetector> pDetector = SimpleBlobDetector::create(params);

        // Loop while there are frames
        while (true) {
            capture >> frame;
            if (frame.empty())
                break;

            Mat roi(frame, Rect(xo, yo, width, height));
            Mat mask = Mat::zeros(roi.size(), CV_8U);
            Point circleCenter(mask.cols / 2, mask.rows / 2);
            int radius = 1300;
            circle(mask, circleCenter, radius, CV_RGB(255, 255, 255), FILLED);
            Mat imagePart = Mat::zeros(roi.size(), roi.type());
            roi.copyTo(imagePart, mask);

            pBackSub->apply(imagePart, fgMask); // outputs 8-bit binary image (0 if black, 255 if white)

            if (framenum) // Ignore first frame
            {

                double min, max;
                //cv::minMaxLoc(fgMask, &min, &max);

                // Sum foreground mask values
                double t = cv::sum(fgMask)[0];

                // If frame has movement
                if (t > 0)
                {
                    sumChannelZero += t;

                    // invert foreground mask then run distance transform
                    bitwise_not(fgMask, dist);
                    //distanceTransform(dist, dist, DIST_L2, 0);

                  


                    // sum distance values (higher indicates lower distribution of pixels)
                    double distribution = sum(dist)[0];
                    //sumDistribution += distribution;

                    // find  maximum distance is (higher value indicates less spread out - not ideal) 
                    //double min, max;
                    //cv::minMaxLoc(dist, &min, &max);
                    //cout << "Max: " << max << endl;

                    // save values if motion is 
                    if (t > sumMax)
                    {
                        sumMax = t;
                        sumMaxTime = capture.get(CAP_PROP_POS_MSEC);

                        std::vector<KeyPoint> keypoints;
                        pDetector->detect(dist, keypoints);

                        Mat im_with_keypoints;
                        drawKeypoints(dist, keypoints, im_with_keypoints, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                        resize(im_with_keypoints, im_with_keypoints, Size(800, 800), 0, 0, INTER_CUBIC);
                        imshow("keypoints", im_with_keypoints);

                       /* Mat hist;
                        int histSize = 2480;
                        float range[] = { 0, 2480 };
                        const float* histRange = { range };
                        calcHist(&dist, 1, 0, Mat(), hist, 1, &histSize, &histRange, 1, 0);

                        hist /= dist.total();
                        hist += 1e-4; //prevent 0

                        Mat logP;
                        cv::log(hist, logP);

                        float entropy = -1 * sum(hist.mul(logP)).val[0];
                        entropyMax += entropy;

                        cout << "Entropy: " << entropy << endl;

                        cout << "image1 row: 0~2 = " << endl << " " << hist.rowRange(0, 2) << endl << endl;*/

                        minMaxLoc(fgMask, &minVal, &maxVal, &minIdx, &maxIdx);
                        maxIdx.x += xo;
                        maxIdx.y += yo;

                        saveFrame = frame;
                        saveMask = fgMask;
                    }

                    motion_frames_count++;
                }
            }


            framenum++;

            resize(imagePart, imagePart, Size(800, 800), 0, 0, INTER_CUBIC);
            imshow("roi", imagePart);

            // Show video frame and foreground mask
            resize(frame, frame, Size(800, 800), 0, 0, INTER_CUBIC);
            imshow("Frame", frame);

            resize(fgMask, fgMask, Size(800, 800), 0, 0, INTER_CUBIC);
            imshow("FG Mask", fgMask);

            //get the input from the keyboard
            int keyboard = waitKey(30);
            if (keyboard == 'q' || keyboard == 27)
                break;

        }

        double motionmetric = sumChannelZero / motion_frames_count;
        double distmetric = (sumDistribution / motion_frames_count) * 0.0001;

        cout << "Metric 1 (Motion): " << motionmetric << endl;
        cout << "Metric 2 (Distribution): " << distmetric << endl;
        cout << "Metric 3 (Entopy): " << entropyMax / motion_frames_count << endl;
        cout << "Time of maximum motion (S): " << sumMaxTime / 1000 << endl;
        cout << "Position of maximum motion: " << maxIdx << endl;

        capture.release();
        pBackSub.release();

        bool save = motionmetric > MOTIONTHRESH && distmetric > DISTTHRESH;
        
        if (save)
        {
            string src = entry.path().string();
            experimental::filesystem::path dest("C:\\Users\\JDMoore_Home\\Desktop\\checknow\\" + entry.path().filename().string());

            try {
                experimental::filesystem::rename(src, dest);
            }
            catch (experimental::filesystem::filesystem_error& e) {
                cout << e.what() << '\n';
            }

            string name = entry.path().filename().string();
            string minusfileext = name.substr(0, name.size() - 4);
            string framepngname = "C:\\Users\\JDMoore_Home\\Desktop\\checknow\\frame_" + minusfileext + ".png";
            string maskpngname = "C:\\Users\\JDMoore_Home\\Desktop\\checknow\\mask_" + minusfileext + ".png";

            if (!saveFrame.empty())
            {
                imwrite(framepngname, saveFrame);
                imwrite(maskpngname, saveMask);
            }

            std::cout << "Saved for checking" << std::endl << std::endl;
        }
        else {

            string src = entry.path().string();
            std::experimental::filesystem::path dest("C:\\Users\\JDMoore_Home\\Desktop\\nomotion\\" + entry.path().filename().string());

            try {
                experimental::filesystem::rename(src, dest);
            }
            catch (experimental::filesystem::filesystem_error& e) {
                cout << e.what() << '\n';
            }

            string name = entry.path().filename().string();
            string minusfileext = name.substr(0, name.size() - 4);
            string framepngname = "C:\\Users\\JDMoore_Home\\Desktop\\nomotion\\frame_" + minusfileext + ".png";
            string maskpngname = "C:\\Users\\JDMoore_Home\\Desktop\\nomotion\\mask_" + minusfileext + ".png";

            if (!saveFrame.empty())
            {
                imwrite(framepngname, saveFrame);
                imwrite(maskpngname, saveMask);
            }

            cout << "Not saved" << std::endl ;
        }

        if (!writeCsvFile(csvFile, entry.path().filename().string(), videoDuration, motionmetric, distmetric, sumMaxTime / 1000, save)) {
            cerr << "Failed to write to file: " << csvFile << "\n";
        }


        processingTimer.stop();
        cout << "Total time: " << processingTimer.getTimeSec() << endl << std::endl;
        processingTimer.reset();
    }

    return 0;
}
