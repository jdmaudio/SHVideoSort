/**
 * @file bg_sub.cpp
 * @brief algorithm for sorting fisheye videos based on motion metrics from Background subtraction
 * @author David Moore
 */

#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <mutex>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>

using namespace cv;
using namespace cv::cuda;
using namespace std;

const string PATHTOVIDEOS = "../videos";

int main(int argc, char* argv[])
{   
	printShortCudaDeviceInfo(getDevice());
	int cuda_devices_number = getCudaEnabledDeviceCount();
	cout << "CUDA Device(s) Number: " << cuda_devices_number << endl;
	DeviceInfo _deviceInfo;
	bool _isd_evice_compatible = _deviceInfo.isCompatible();
	cout << "CUDA Device(s) Compatible: " << _isd_evice_compatible << endl;
	
	// GPU Loop
	TickMeter processingTimer;

    for (const auto& entry : experimental::filesystem::directory_iterator(PATHTOVIDEOS)) 
    {
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
		cout << "Aspect Ratio: " << aspectRatio << endl << endl;
		
		Mat frame, fgmask;
		capture >> frame;
		GpuMat d_frame(frame);
		GpuMat d_fgmask;

		Ptr<BackgroundSubtractor> pBackSubGPU = cuda::createBackgroundSubtractorMOG2(500, 16, false);
		pBackSubGPU->apply(d_frame, d_fgmask);

		namedWindow("image", WINDOW_NORMAL);
		namedWindow("foreground mask", WINDOW_NORMAL);

        // Frame loop
		processingTimer.start();
		for (;;)
		{
			capture >> frame;
			if (frame.empty())
				break;
			d_frame.upload(frame);

			int64 start = cv::getTickCount();

			pBackSubGPU->apply(d_frame, d_fgmask);

			double frameSum = cuda::sum(d_fgmask)[0];

			double fps = cv::getTickFrequency() / (cv::getTickCount() - start);
			//std::cout << "FPS : " << fps << std::endl;

			d_fgmask.download(fgmask);

			imshow("image", frame);
			imshow("foreground mask", fgmask);

			char key = (char)waitKey(30);
			if (key == 27)
				break;
		}
		processingTimer.stop();
        capture.release();
        pBackSubGPU.release();

		cout << "GPU Total time: " << processingTimer.getTimeSec() << endl << endl;
	}

	// CPU Loop
	for (const auto& entry : experimental::filesystem::directory_iterator(PATHTOVIDEOS))
	{
		VideoCapture capture(entry.path().string());
		if (!capture.isOpened()) {
			cerr << "Unable to open: " << entry.path().string() << endl;
			return 0;
		}
		cout << entry.path() << endl;

		Mat frame;
		capture >> frame;

		Ptr<BackgroundSubtractor> pBackSubCPU = cv::createBackgroundSubtractorMOG2(500,16,false);

		Mat fgmask;

		pBackSubCPU->apply(frame, fgmask);

		// Frame loop
		processingTimer.reset();
		processingTimer.start();
		for (;;)
		{
			capture >> frame;
			if (frame.empty())
				break;

			int64 start = cv::getTickCount();

			pBackSubCPU->apply(frame, fgmask);

			double frameSum = cv::sum(fgmask)[0];

			double fps = cv::getTickFrequency() / (cv::getTickCount() - start);
			//std::cout << "FPS : " << fps << std::endl;

			imshow("image", frame);
			imshow("foreground mask", fgmask);

			char key = (char)waitKey(30);
			if (key == 27)
				break;
		}

		processingTimer.stop();
		capture.release();
		pBackSubCPU.release();

		cout << "CPU Total time: " << processingTimer.getTimeSec();
	}

    return 0;
}
