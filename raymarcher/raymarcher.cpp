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
#include "transpiler.h"

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
	int chunk = 0;

	for (int i = 1; i < argc; i++)
	{
		int tmp;
		float tx, ty, tz;

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
		else if (sscanf(argv[i], "-p=%f,%f,%f", &tx, &ty, &tz) == 3)
		{
			px = tx;
			py = ty;
			pz = tz;
		}
		else if (sscanf(argv[i], "-g=%f,%f,%f", &tx, &ty, &tz) == 3)
		{
			gx = tx;
			gy = ty;
			gz = tz;
		}
		else if (strlen(argv[i]) == strlen("-k") && strncmp(argv[i], "-k", strlen("-k")) == 0)
		{
			keepOpen = true;
		}
		else if (sscanf(argv[i], "-c=%u", &chunk))
		{
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
			buffer = new char[len + 1]{ '\0' };
			fread(buffer, sizeof(char), len, fp);
			fclose(fp);
		}
		else
		{
			printf("Could not open file %s\n", sfname);
			cuCtxDestroy(context);
			return -1;
		}

		Transpiler transpiler(buffer);
		if (!transpiler.Transpile(cuDevice, cuModule, loweredName))
		{
			printf("An error occured during transpilation.\n");
			cuCtxDestroy(context);
			return -1;
		}
	}
	else
	{
		printf("No input scene file\n");
		cuCtxDestroy(context);
		return -1;
	}

	Vec position(px, py, pz);
	Vec goal = !(Vec(gx, gy, gz) - position);
	Vec left = !Vec(goal.Z(), 0, -goal.X()) / width;
	Vec up = goal ^ left;

	int sx = 0;
	int sy = 0;
	int ex, ey;
	if (chunk)
	{
		ex = width < chunk ? width : chunk;
		ey = height < chunk ? height : chunk	;
	}
	else
	{
		ex = width;
		ey = height;
	}

	CUfunction render;
	cuModuleGetFunction(&render, cuModule, loweredName); // mangled name?

	CUdeviceptr deviceCols;
	cuMemAlloc(&deviceCols, width * height * sizeof(Vec));
	void* args[] = { &deviceCols, &width, &height, &samples, &position, &goal, &left, &up, &sx, &sy, &ex, &ey };

	CUevent start, stop;
	cuEventCreate(&start, CU_EVENT_DEFAULT);
	cuEventCreate(&stop, CU_EVENT_DEFAULT);

	Vec* cols = new Vec[width * height];

	cuEventRecord(start, nullptr);

	if (chunk)
	{
		CUstream streams[8];
		for (int i = 0; i < sizeof(streams) / sizeof(*streams); i++)
		{
			cuStreamCreate(&streams[i], CU_STREAM_DEFAULT);
		}
		int streamIndex = 0;
		do
		{
			do
			{
				cuLaunchKernel(render, 256, 256, 1, 16, 16, 1, 0, streams[streamIndex++], args, nullptr);
				sy += chunk;
				ey += chunk;

				if (streamIndex == sizeof(streams) / sizeof(*streams))
				{
					streamIndex = 0;
					cuCtxSynchronize();
				}
			}
			while (ey <= height); // misses top row

			// render top row
			cuLaunchKernel(render, 256, 256, 1, 16, 16, 1, 0, streams[streamIndex++], args, nullptr);

			if (streamIndex == sizeof(streams) / sizeof(*streams))
			{
				streamIndex = 0;
				cuCtxSynchronize();
			}

			sx += chunk;
			ex += chunk;
			sy = 0;
			ey = height < chunk ? height : chunk;
		}
		while (ex <= width);

		for (int i = 0; i < sizeof(streams) / sizeof(*streams); i++)
		{
			cuStreamDestroy(streams[i]);
		}
	}
	else
	{
		cuLaunchKernel(render, 256, 256, 1, 16, 16, 1, 0, nullptr, args, nullptr);
	}

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

	if (keepOpen)
	{
		printf("Press any key to exit...\n");
		getchar();
	}

	return 0;
}