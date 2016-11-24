/*
Digit Recognition, author : greg smith.
uses Support Vector Machines to train an predict handwritten digits.
*/

#include <opencv2\opencv.hpp>
#include <stdio.h>
#include <vector>
#include <fstream>

#define NUM_TRAINING_IMAGES 60000
#define TRAINING_IMAGES_WIDTH 28
#define TRAINING_IMAGES_HEIGHT 28
#define TRAINING_SET_IMAGE_PATH "../Training Set/train-images.idx3-ubyte"
#define TRAINING_SET_LABEL_PATH "../Training Set/train-labels.idx1-ubyte"

#define NUM_TEST_IMAGES 10000
#define TEST_SET_IMAGE_PATH "../Training Set/t10k-images.idx3-ubyte"
#define TEST_SET_LABEL_PATH "../Training Set/t10k-labels.idx1-ubyte"

#define EXPORT_PATH "../Exports/HandwritingSVM.bin"

using namespace std;
using namespace cv;

/*
main : main entry point.
@param int argc : number of test images passed.
@param char* argv[] : array of strings. Used to pass the file paths of handwritten digits to be tested.
*/
int main(int argc, char* argv[]);

/*
void loadTrainingSet : loads the training set of handwritten digits and places the images in 'trainingImages' and the lables in 'trainingLabels'.
						the training set comes from the MNIST databse and contains 60000 uniqe 28x28 handwritten digits.
@param imageFilePath : specifies the filepath of the image training set.
@param labelFilePath : specifices the filepath of the label training set.
@param trainingSetSize : specifies the size of the training set.
@param trainingImages : vector used to contain the training images.
@param trainingLabels : vector used to contain the training labels.
*/
void loadTrainingSet(char* imageFilePath, char* labelFilePath, int trainingSetSize, vector<Mat*> &trainingImages, vector<unsigned char> &trainingLabels);

/*
void exportTrainingImages : writes the images in the training set to the disk as a series of png's.
@param filePath : specifies the folder to write the images.
@param trainingImages : vector which contains the training images.
@param trainingLabels : vector which contains the training labels.
*/
void exportTrainingImages(char* filePath, vector<Mat*> &trainingImages, vector<unsigned char> &trainingLabels);

/*
CvSVM* trainSVM : trains a support vector machine on the training images and labels.
@param trainingImages : vector which contains the training images.
@param trainingLabels : vector which contains the training labels.

@return CvSVM* : returns a pointer to the newly created support vector machine object.
*/
CvSVM* trainSVM(vector<Mat*> &trainingImages, vector<unsigned char> &trainingLabels);

/*
unsigned char predictLabel : predicts the digit of the input image 'image'
@param image : test example handwritten image.
@param svm : support vector machine object pointer.

return unsigned char: returns the digit prediction [0 - 9]
*/
unsigned char predictLabel(Mat* image, CvSVM* svm);

/*
float testSVMError : tests the algorithm on the test set. returns the overall error rate and the error rate for every digit.
@param svm : Support Vector Machine object pointer.
@param errorDigitRate : floating point array of size 10. Used to store the error of each digit.

@return float : error rate.
*/
float testSVMError(CvSVM* svm, float* errorDigitRate);

int main(int argc, char* argv[])
{
	CvSVM* svm;
#ifdef DIGIT_TRAIN
	vector<Mat*> trainingImages;
	vector<unsigned char> trainingLabels;
	
	/*load the training set*/
	loadTrainingSet(TRAINING_SET_IMAGE_PATH, TRAINING_SET_LABEL_PATH, NUM_TRAINING_IMAGES, trainingImages, trainingLabels);
	
	/*traing the support vector machine*/
	svm = trainSVM(trainingImages, trainingLabels);
	
	/*save the SVM object to the disk*/
	svm->save(EXPORT_PATH);

	trainingImages.clear();
	trainingLabels.clear();

#endif


#ifdef DIGIT_PREDICT
	svm = new CvSVM();

	/*load the SVM object from the disk*/
	svm->load(EXPORT_PATH);

	if (argc <= 1)
	{
		printf("Error, not enough parameters");
	}
	else
	{
		int numTestImages = argc - 1;
		Mat imageLoad;
		unsigned char label;

		std::ofstream outputFile;
		outputFile.open("Handwriting Class Results.txt", std::ofstream::out);

		ofstream imageFile;


		for (int imageIndex = 1; imageIndex < argc; imageIndex++)
		{
			/*get the input image file path*/
			char* impath = argv[imageIndex];

			imageFile.open(impath, std::ofstream::in);
			/*check if the file exists*/
			if (imageFile.is_open())
			{
				/*load the image*/
				imageLoad = imread(impath, CV_LOAD_IMAGE_GRAYSCALE);
				/*predict*/
				label = predictLabel(&imageLoad, svm);

				/*export prediction to a file*/
				outputFile << impath;
				outputFile << " : ";
				outputFile << to_string(label);
				outputFile << "\n";
			}
			else
			{
				printf("Error: cannot find file : ");
				printf(impath);
				printf("\n");
			}
			imageFile.close();
		}

		outputFile.close();
	}

	system("pause");

#endif
	
	delete svm;
}

unsigned char predictLabel(Mat* image, CvSVM* svm)
{
	Mat sampleMat(1, TRAINING_IMAGES_WIDTH*TRAINING_IMAGES_HEIGHT, CV_32FC1);
	int pixelIndex = 0;
	for (int x = 0; x < TRAINING_IMAGES_WIDTH; x++)
	{
		for (int y = 0; y < TRAINING_IMAGES_HEIGHT; y++)
		{
			float imageSample = (image->at<unsigned char>(x, y));
			sampleMat.at<float>(0, pixelIndex) = imageSample;
			pixelIndex++;
		}
	}
	float response = svm->predict(sampleMat);

	unsigned char respChar = round(response);
	return respChar;
}

CvSVM* trainSVM(vector<Mat*> &trainingImages, vector<unsigned char> &trainingLabels)
{
	Mat labelsMat(trainingLabels.size(), 1, CV_32FC1);
	for (int i = 0; i < trainingLabels.size(); i++)
	{
		float labelSample = (float)trainingLabels[i];
		labelsMat.at<float>(i, 0) = labelSample;
	}


	Mat trainingDataMat(trainingImages.size(), TRAINING_IMAGES_WIDTH*TRAINING_IMAGES_HEIGHT, CV_32FC1);
	for (int i = 0; i < trainingImages.size(); i++)
	{
		int pixelIndex = 0;
		for (int x = 0; x < TRAINING_IMAGES_WIDTH; x++)
		{
			for (int y = 0; y < TRAINING_IMAGES_HEIGHT; y++)
			{
				float imageSample = (trainingImages[i]->at<unsigned char>(x, y));
				trainingDataMat.at<float>(i, pixelIndex) = imageSample;
				pixelIndex++;
			}
		}

	}

	CvSVMParams params = CvSVMParams::CvSVMParams();
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::POLY;
	params.gamma = 1;
	params.coef0 = 0;
	params.degree = 3;

	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

	CvSVM* svm = new CvSVM();
	Mat varIdx;
	Mat sampleIdx;
	svm->train(trainingDataMat, labelsMat, varIdx, sampleIdx, params);

	trainingDataMat.release();
	labelsMat.release();

	return svm;
}

float testSVMError(CvSVM* svm, float* errorDigitRate)
{
	vector<Mat*> testImages;
	vector<unsigned char> testLabels;
	loadTrainingSet(TEST_SET_IMAGE_PATH, TEST_SET_LABEL_PATH, NUM_TEST_IMAGES, testImages, testLabels);

	int numDigits[10];
	unsigned int numCorrectDigit[10];
	float errorRate = 0.0;
	unsigned int numCorrect = 0;
	for (int i = 0; i < 10; i++)
	{
		errorDigitRate[i] = 0.0;
		numCorrectDigit[i] = 0;
		numDigits[i] = 0;
	}


	unsigned int total = testImages.size();
	for (int i = 0; i < testImages.size(); i++)
	{
		unsigned char label = predictLabel(testImages[i], svm);
		unsigned char truth = testLabels[i];
		bool correct = label == truth;
		
		if (correct)
		{
			numCorrect++;
			numCorrectDigit[label]++;
		}
	
		numDigits[label]++;

		
	}
	
	errorRate = 1.0 - (float)numCorrect / (float)total;
	for (int i = 0; i < 10; i++)
	{
		errorDigitRate[i] = 1.0 - (float)numCorrectDigit[i] / (float)numDigits[i];
	}

	return errorRate;
}

void loadTrainingSet(char* imageFilePath, char* labelFilePath, int trainingSetSize, vector<Mat*> &trainingImages, vector<unsigned char> &trainingLabels)
{
	fstream imageFile;
	fstream labelFile;
	
	imageFile.open(imageFilePath, std::fstream::in | std::fstream::binary);
	labelFile.open(labelFilePath, std::fstream::in | std::fstream::binary);
	
	int imageHeader1, imageHeader2, imageHeader3, imageHeader4;
	int labelHeader1, labelHeader2;

	imageFile.read((char*)&imageHeader1, 4);
	imageFile.read((char*)&imageHeader2, 4);
	imageFile.read((char*)&imageHeader3, 4);
	imageFile.read((char*)&imageHeader4, 4);

	labelFile.read((char*)&labelHeader1, 4);
	labelFile.read((char*)&labelHeader2, 4);

	unsigned char pixel;
	unsigned char label;
	Mat* image;

	for (int numImages = 0; numImages < trainingSetSize; numImages++)
	{
		image = new Mat(TRAINING_IMAGES_WIDTH, TRAINING_IMAGES_HEIGHT, CV_8U);
		for (int y = 0; y < TRAINING_IMAGES_WIDTH; y++)
		{
			for (int x = 0; x < TRAINING_IMAGES_HEIGHT; x++)
			{
				imageFile.read((char*)&pixel, 1);
				image->at<unsigned char>(y, x) = pixel;
			}
		}
		trainingImages.push_back(image);

		labelFile.read((char*)&label, 1);
		trainingLabels.push_back(label);
	}
	
	imageFile.close();
	labelFile.close();

}


void exportTrainingImages(char* filePath, vector<Mat*> &trainingImages, vector<unsigned char> &trainingLabels)
{
	Mat* image;
	for (int i = 0; i < trainingImages.size(); i++)
	{
		image = trainingImages[i];
		string imagesName = filePath;
		imagesName.append("/trainingImage_index");
		imagesName.append(to_string(i));
		imagesName.append("_label");
		imagesName.append(to_string(trainingLabels[i]));
		imagesName.append(".png");
		imwrite(imagesName, *image);
	}
}