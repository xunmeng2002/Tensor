#include "TestTensor.h"

using namespace std;


int main()
{
	DeviceType device = DeviceType::CPU;
	//DeviceType device = DeviceType::CUDA;
	TestTensor testTensor(device);
	testTensor.Test();


    return 0;
}
