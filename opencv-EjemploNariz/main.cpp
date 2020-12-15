#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>


using namespace std;
using namespace cv;


void detectAndDisplay(Mat frame);
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
CascadeClassifier smile_cascade;
CascadeClassifier nose_cascade;

int main(int argc, const char** argv)
{
    CommandLineParser parser(argc, argv,
        "{help h||}"
        "{face_cascade|data/haarcascades/haarcascade_frontalface_alt.xml|Path to face cascade.}"
        "{eyes_cascade|data/haarcascades/haarcascade_eye_tree_eyeglasses.xml|Path to eyes cascade.}"
        "{smile_cascade|data/haarcascades/haarcascade_smile.xml|Path to smile cascade.}"
        "{nose_cascade|data/haarcascades/haarcascade_nose.xml|Path to smile cascade.}"
        "{camera|0|Camera device number.}");
    parser.about("\nThis program demonstrates using the cv::CascadeClassifier class to detect objects (Face + eyes) in a video stream.\n"
        "You can use Haar or LBP features.\n\n");
    parser.printMessage();
    String face_cascade_name = samples::findFile(parser.get<String>("face_cascade"));
    String eyes_cascade_name = samples::findFile(parser.get<String>("eyes_cascade"));
    String smile_cascade_name = samples::findFile(parser.get<String>("smile_cascade"));
    String nose_cascade_name = samples::findFile(parser.get<String>("nose_cascade"));

    //-- 1. Load the cascades
    if (!face_cascade.load(face_cascade_name))
    {
        cout << "--(!)Error loading face cascade\n";
        return -1;
    };
    if (!eyes_cascade.load(eyes_cascade_name))
    {
        cout << "--(!)Error loading eyes cascade\n";
        return -1;
    };
    if (!smile_cascade.load(smile_cascade_name))
    {
        cout << "--(!)Error loading boca cascade\n";
        return -1;
    };

    if (!nose_cascade.load(nose_cascade_name))
    {
        cout << "--(!)Error loading nariz cascade\n";
        return -1;
    };
    int camera_device = parser.get<int>("camera");
    VideoCapture capture;
    //-- 2. Read the video stream
    capture.open(camera_device);
    if (!capture.isOpened())
    {
        cout << "--(!)Error opening video capture\n";
        return -1;
    }
    Mat frame;
    while (capture.read(frame))
    {
        if (frame.empty())
        {
            cout << "--(!) No captured frame -- Break!\n";
            break;
        }
        //-- 3. Apply the classifier to the frame
        detectAndDisplay(frame);
        if (waitKey(10) == 27)
        {
            break; // escape

        }
    }
    return 0;
}
void detectAndDisplay(Mat frame)
{
    Mat frame_gray;
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY,COLOR_BGR5652RGBA);
    equalizeHist(frame_gray, frame_gray);
    
    //-- Detect faces
    std::vector<Rect> faces;
    face_cascade.detectMultiScale(frame_gray, faces);
    for (size_t i = 0; i < faces.size(); i++)
    {
        Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
        ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4);
        Mat faceROI = frame_gray(faces[i]);

        //-- In each face, detect eyes
        vector<Rect> eyes;
        eyes_cascade.detectMultiScale(faceROI, eyes);
        for (size_t j = 0; j < eyes.size(); j++)
        {
            Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
            int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
            circle(frame, eye_center, radius, Scalar(255, 0, 0), 4);
        }

        std::vector<Rect> smile;
        smile_cascade.detectMultiScale(faceROI, smile, 1.1, 2, 0, Size(80, 80));
        for (size_t k = 0; k < smile.size(); k++)
        {
            Point smile_center(faces[i].x + smile[k].x + smile[k].width / 2, faces[i].y + smile[k].y + smile[k].height / 2);
            int radius = cvRound(( smile[k].width  + smile[k].height) * 0.25);
            circle(frame, smile_center, radius, Scalar(0, 0, 255), 4, 8, 0);
        }

        vector<Rect> nose;
        nose_cascade.detectMultiScale(faceROI, nose, 1.1, 2, 0, Size(40, 40));
        for (size_t k = 0; k < nose.size(); k++)
        {
            Point nose_center(faces[i].x + nose[k].x + nose[k].width / 2, faces[i].y + nose[k].y + nose[k].height / 3);
            int radius = cvRound((nose[k].width + nose[k].height) * 0.25);
            circle(frame, nose_center, radius, Scalar(0, 0, 255), 4, 8, 4);
        }
        //-- Show what you got
        imshow("Capture - Face detection", frame);
    }
}
