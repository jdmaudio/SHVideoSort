/**
 * @file bg_sub.cpp
 * @brief algorithm for sorting fisheye videos based on motion metrics from Background subtraction
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

const double MOTIONTHRESH = 2300;
const double BLOBDIATHRESH = 10;
const string PATHTOVIDEOS = "../videos";
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
	string csvFile = "results.csv";

    if (!fileExists(csvFile))
        writeCsvFile(csvFile, "Filename", "Duration", "Metric 1", "Metric 2", "Time of max", "Saved (0 or 1)");

	Ptr<BackgroundSubtractor> pBackSub;
	TickMeter processingTimer;

    // Iterate over video files
    for (const auto& entry : experimental::filesystem::directory_iterator(PATHTOVIDEOS)) 
    {
        processingTimer.start(); // Record processing time

        VideoCapture capture(entry.path().string());
        if (!capture.isOpened()) {
            cerr << "Unable to open: " << entry.path().string() << endl;
            return 0;
        }
        cout << entry.path() << endl;

        double videoDuration = capture.get(CAP_PROP_FRAME_COUNT) / capture.get(CAP_PROP_FPS);
        cout << "Duration (seconds): " << videoDuration << endl;

		double videoWidth = capture.get(CAP_PROP_FRAME_WIDTH);
		double videoHeight = capture.get(CAP_PROP_FRAME_HEIGHT);
		double aspectRatio = videoWidth / videoHeight;
		cout << "Width (px): " << videoWidth << endl;
		cout << "Height (px): " << videoHeight << endl;
		cout << "Aspect Ratio: " << aspectRatio << endl;

        double sumChannelZero = 0.0;
        double sumDistribution = 0.0;
        double sumMax = 0.0;
        double maxMotionTime = 0.0;
        Point minIdx, maxIdx;

        Mat frame, fgMask, saveFrame, saveMask, saveBlob, dist;
        int frameNum = 0;
        int motionFramesCount = 0;

        // Region of interest (assumes all videos are 2880x2880)
        int xo = 200, yo = 200;  // int xo = 760, yo = 260;
        int width = 2480, height = 2480;

        pBackSub = createBackgroundSubtractorMOG2(25, 500, false);

        // Setup SimpleBlobDetector parameters.
        SimpleBlobDetector::Params params;
        params.filterByArea = true;
        params.minArea = 50;
        params.maxArea = 10000;
		params.thresholdStep = 20;
        params.filterByCircularity = false;
        params.filterByConvexity = false;
        params.filterByInertia = false;

        Ptr<SimpleBlobDetector> pBlobDetector = SimpleBlobDetector::create(params);
		std::vector<KeyPoint> keyPointsSave;

		// Get first frame
		capture >> frame;
		if (frame.empty())
			break;

		// Create a region of interest and circular mask
		int maxRadius = 1300;
		Mat roi(frame, Rect(xo, yo, width, height)); 
		Mat mask = Mat::zeros(roi.size(), CV_8U); 
		Point circleCenter(mask.cols / 2, mask.rows / 2); 
		circle(mask, circleCenter, maxRadius, CV_RGB(255, 255, 255), FILLED); 
		Mat imagePart = Mat::zeros(roi.size(), roi.type());
		roi.copyTo(imagePart, mask);

		// Run backsubtraction on first frame - outputs 8-bit binary image (0 if black, 255 if white)
		pBackSub->apply(imagePart, fgMask);
		
        // Frame loop
        while (true) {

            capture >> frame;
            if (frame.empty())
                break;
			
			roi = frame(Rect(xo, yo, width, height));
			roi.copyTo(imagePart, mask); // Apply mask

            pBackSub->apply(imagePart, fgMask); 

            // Sum foreground mask values
            double t = cv::sum(fgMask)[0];

            // If frame has any movement
            if (t > 0)
            {
                sumChannelZero += t;
                
				// Invert mask for blob detector
				bitwise_not(fgMask, dist);

                // save values at maximum motion
                if (t > sumMax)
                {
                    sumMax = t;
					maxMotionTime = capture.get(CAP_PROP_POS_MSEC);

                    std::vector<KeyPoint> keypoints;
					pBlobDetector->detect(dist, keypoints);

                    Mat imageWithKeypoints;
                    drawKeypoints(dist, keypoints, imageWithKeypoints, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                    saveFrame = frame;
                    saveMask = fgMask;
					saveBlob = imageWithKeypoints;
					keyPointsSave = keypoints;
                }

                motionFramesCount++;
            }
            
			if (!saveBlob.empty()) 
			{
				resize(saveBlob, saveBlob, Size(800, 800), 0, 0, INTER_CUBIC);
				imshow("keypoints", saveBlob);
			}

			if (!frame.empty())
			{
				resize(frame, frame, Size(800, 800), 0, 0, INTER_CUBIC);
				imshow("Frame", frame);
			}

			if (!fgMask.empty())
			{
				resize(fgMask, fgMask, Size(800, 800), 0, 0, INTER_CUBIC);
				imshow("FG Mask", fgMask);
			}

            //get the input from the keyboard
            int keyboard = waitKey(30);
            if (keyboard == 'q' || keyboard == 27)
                break;

        }

        double motionMetric = sumChannelZero / motionFramesCount;
		
		double maxBlobSize = 0.0;
		for (KeyPoint kp : keyPointsSave) {

			if(kp.size > maxBlobSize)
				maxBlobSize = kp.size;
		}

        cout << "Metric 1 (Motion): " << motionMetric << endl;
		cout << "Metric 2 (Blob size): " << maxBlobSize << endl;
        cout << "Time of maximum motion (S): " << maxMotionTime / 1000 << endl;

        capture.release();
        pBackSub.release();

        bool save = motionMetric > MOTIONTHRESH && maxBlobSize > BLOBDIATHRESH;
        
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

            cout << "Saved for checking" << endl << endl;
        }
        else {

            string src = entry.path().string();
            experimental::filesystem::path dest("C:\\Users\\JDMoore_Home\\Desktop\\nomotion\\" + entry.path().filename().string());

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

        if (!writeCsvFile(csvFile, entry.path().filename().string(), videoDuration, motionMetric, maxBlobSize, maxMotionTime / 1000, save)) {
            cerr << "Failed to write to file: " << csvFile << "\n";
        }

        processingTimer.stop();
        cout << "Total time: " << processingTimer.getTimeSec() << endl << endl;
        processingTimer.reset();
    }

    return 0;
}
