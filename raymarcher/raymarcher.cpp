#define _CRT_SECURE_NO_WARNINGS

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <cmath>
#include <cstdint>
#include <cstdio>

#define RPNG_IMPLEMENTATION
#include "rpng.h"

#include "utils.h"

#define VALID_FILENAME_CHARS "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890._"
#define VALID_FILEPATH_CHARS ":/\\"

int main(int argc, char** argv)
{
	int width = 160;
	int height = 90;
	float px = 1.0f, py = 1.0f, pz = 1.0f, gx = 0.0f, gy = 0.0f, gz = 0.0f;
	int samples = 25;
	char fname[256] = "out.png";
	char sfname[256] = "";

	bool keepOpen = false;

	for (int i = 1; i < argc; i++)
	{
		int tmp;

		if (sscanf(argv[i], "-w=%u", &tmp) && tmp > 0)
		{
			width = tmp;
		}
		else if (sscanf(argv[i], "-h=%u", &tmp) && tmp > 0)
		{
			height = tmp;
		}
		else if (sscanf(argv[i], "-s=%u", &tmp) && tmp > 0)
		{
			samples = tmp;
		}
		else if (sscanf(argv[i], "-px=%f", &px))
		{
		}
		else if (sscanf(argv[i], "-py=%f", &py))
		{
		}
		else if (sscanf(argv[i], "-pz=%f", &pz))
		{
		}
		else if (sscanf(argv[i], "-gx=%f", &gx))
		{
		}
		else if (sscanf(argv[i], "-gy=%f", &gy))
		{
		}
		else if (sscanf(argv[i], "-gz=%f", &gz))
		{
		}
		else if (sscanf(argv[i], "-k"))
		{
			keepOpen = true;
		}
		else if (sscanf(argv[i], "-f=%255[" VALID_FILENAME_CHARS "]", fname))
		{
		}
		else if (sscanf(argv[i], "-sf=%255[-" VALID_FILENAME_CHARS VALID_FILEPATH_CHARS "]", sfname))
		{
		}
		else
		{
			printf("Unknown or invalid argument '%s'\n", argv[i]);
		}
	}

	char* log = nullptr;
	char* ptx = nullptr;
	const char* loweredName;
	nvrtcProgram prog;

	CUdevice cuDevice;
	CUcontext context;
	CUmodule cuModule;
	cuInit(0);
	cuDeviceGet(&cuDevice, 0);
	cuCtxCreate(&context, 0, cuDevice);

	if (strlen(sfname) > 0)
	{
		FILE* fp = fopen(sfname, "rb");
		char* buffer = nullptr;
		if (fp)
		{
			fseek(fp, 0, SEEK_END);
			size_t len = ftell(fp);
			fseek(fp, 0, SEEK_SET);
			buffer = new char[len];
			fread(buffer, sizeof(char), len, fp);
			fclose(fp);
		}
		else
		{
			printf("Could not open file %s", sfname);
			cuCtxDestroy(context);
			return -1;
		}

		nvrtcCreateProgram(&prog, buffer, "QuerySceneUserFn", 0, nullptr, nullptr);

		nvrtcAddNameExpression(prog, "Render");

		char versionOption[20]{ '\0' };

		int major;
		int minor;
		cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice);
		cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice);

		sprintf(versionOption, "-arch=sm_%d%d", major, minor);
		const char* opts[] = { "-std=c++14", "-I.", "-I../raymarcher", versionOption };
		printf("sm_%d.%d\n", major, minor);
		nvrtcResult result = nvrtcCompileProgram(prog, sizeof(opts) / sizeof(*opts), opts); // crashes?

		size_t logSize;
		nvrtcGetProgramLogSize(prog, &logSize);
		log = new char[logSize];
		nvrtcGetProgramLog(prog, log);

		if (result != NVRTC_SUCCESS)
		{
			printf("Invalid ptx code generated from file %s:\n%s", sfname, log);
			nvrtcDestroyProgram(&prog);
			delete[] ptx;
			delete[] log;
			cuCtxDestroy(context);
			return -1;
		}

		size_t ptxSize;
		nvrtcGetPTXSize(prog, &ptxSize);
		ptx = new char[ptxSize];
		nvrtcGetPTX(prog, ptx);

		nvrtcGetLoweredName(prog, "Render", &loweredName);
	}
	else
	{
		printf("No input scene file");
		cuCtxDestroy(context);
		return -1;
	}

	Vec position(px, py, pz);
	Vec goal = !(Vec(gx, gy, gz) - position);
	Vec left = !Vec(goal.Z(), 0, -goal.X()) / width;
	Vec up = goal ^ left;
	
	CUjit_option jitOpts[] = { CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, CU_JIT_LOG_VERBOSE };
	char* errBuf = new char[4096];
	void* arr[] = { errBuf, (void*)4096, (void*)true };
	CUresult err = cuModuleLoadDataEx(&cuModule, ptx, 3, jitOpts, arr);
	if (err != CUDA_SUCCESS)
	{
		printf("Error:\n%s", errBuf);
		delete[] errBuf;
		delete[] ptx;
		delete[] log;
		cuCtxDestroy(context);
		return -1;
	}
	CUfunction render;
	cuModuleGetFunction(&render, cuModule, loweredName); // mangled name?
	delete[] errBuf;

	CUdeviceptr deviceCols;
	cuMemAlloc(&deviceCols, width * height * sizeof(Vec));
	void* args[] = { &deviceCols, &width, &height, &samples, &position, &goal, &left, &up };

	CUevent start, stop;
	cuEventCreate(&start, CU_EVENT_DEFAULT);
	cuEventCreate(&stop, CU_EVENT_DEFAULT);

	Vec* cols = new Vec[width * height];

	cuEventRecord(start, nullptr);
	cuLaunchKernel(render, 256, 256, 1, 16, 16, 1, 0, nullptr, args, nullptr);
	cuEventRecord(stop, nullptr);
	cuMemcpyDtoH(cols, deviceCols, width * height * sizeof(Vec));
	cuEventSynchronize(stop);

	float ms;
	cudaEventElapsedTime(&ms, start, stop);
	printf("Elapsed time: %4f seconds\n", ms / 1000);

	char* data = new char[width * height * 3];

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			Vec color = cols[x + y * width] + (14.0f / 241.0f);
			Vec o = color + 1;
			color /= o;
			color *= 255;

			data[((width - x - 1) + (height - y - 1) * width) * 3] = color.X();
			data[((width - x - 1) + (height - y - 1) * width) * 3 + 1] = color.Y();
			data[((width - x - 1) + (height - y - 1) * width) * 3 + 2] = color.Z();
		}
	}

	delete[] cols;

	rpng_create_image(fname, data, width, height, 3, 8);

	delete[] data;

	delete[] log;
	delete[] ptx;

	cuMemFree(deviceCols);

	cuModuleUnload(cuModule);
	cuCtxDestroy(context);

	nvrtcDestroyProgram(&prog);

	if (keepOpen)
	{
		printf("Press any key to exit...\n");
		getchar();
	}

	return 0;
}