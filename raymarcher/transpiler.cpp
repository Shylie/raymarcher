#define _CRT_SECURE_NO_WARNINGS

#include "transpiler.h"

#include <cstdio>

#include <string>

Transpiler::Transpiler(const char* source) : lexer(source), current{}, previous{}, hadError(false), panicMode(false) {}

bool Transpiler::Transpile(CUdevice& cuDevice, CUmodule& cuModule, const char*& loweredName)
{
	Advance();
	Consume(Token::Type::DeclareMaterials, "Expect decmat statement.");
	do
	{
		Consume(Token::Type::Comma, "Expect comma.");
		Consume(Token::Type::String, "Expect identifier.");
		Mat m;
		m.start = previous.start;
		m.length = previous.length;
		mats.push_back(m);
	} while (current.type == Token::Type::Comma);

	do
	{
		Consume(Token::Type::SDF, "Expect sdf statement.");
		Consume(Token::Type::Comma, "Expect comma.");
		switch (current.type)
		{
		case Token::Type::Box: Box(); break;
		case Token::Type::Sphere: Sphere(); break;
		case Token::Type::Transform: Transform(); break;
		default: ErrorAtCurrent("Expect a valid SDF identifier."); break;
		}

		panicMode = false;
	} while (current.type == Token::Type::SDF);

	do
	{
		Consume(Token::Type::Material, "Expect mat statement.");
		Consume(Token::Type::Comma, "Expect comma.");
		Consume(Token::Type::String, "Expect identifier.");
		const char* matNameStart = previous.start;
		int matNameLength = previous.length;
		int idx = -1;
		for (int i = 0; i < mats.size(); i++)
		{
			if (mats[i].length == matNameLength && strncmp(mats[i].start, matNameStart, matNameLength) == 0)
			{
				idx = i;
				break;
			}
		}
		if (idx == -1)
		{
			Error("Undeclared material name.");
		}
		else
		{
			Consume(Token::Type::Param, "Expect param statement (>).");
			const char* paramStart = current.start;
			MakeMatInternal();
			int paramLength = (int)(previous.start - paramStart) + previous.length;
			mats[idx].paramStart = paramStart;
			mats[idx].paramLength = paramLength;
		}

		panicMode = false;
	}
	while (current.type == Token::Type::Material);

	Consume(Token::Type::End, "Expect end of file.");

	if (hadError) { return false; }

	using namespace std::string_literals;
	/*
	#include "utils.h"
	typedef bool (*MaterialFn)(Vec& origin, Vec& direction, Vec& color, Vec& attenuation, Vec sampledPosition, Vec normal, xorwow& random);
	#define MATERIAL_FUNCTION(name) extern bool name##(Vec& origin, Vec& direction, Vec& color, Vec& attenuation, Vec sampledPosition, Vec normal, xorwow& random)
	MATERIAL_FUNCTION(None);
	*/
	std::string transpiledSource = R"(#include "utils.h"
typedef bool (*MaterialFn)(Vec& origin, Vec& direction, Vec& color, Vec& attenuation, Vec sampledPosition, Vec normal, xorwow& random);
#define MATERIAL_FUNCTION(name) __device__ extern bool name##(Vec& origin, Vec& direction, Vec& color, Vec& attenuation, Vec sampledPosition, Vec normal, xorwow& random);
MATERIAL_FUNCTION(none);
)"s;

	for (int i = 0; i < mats.size(); i++)
	{
		transpiledSource += "MATERIAL_FUNCTION(";
		transpiledSource += std::string(mats[i].start, mats[i].length);
		transpiledSource += ");\n";
	}

	transpiledSource += "__device__ MaterialFn materials[] =\n{\nnone";
	for (int i = 0; i < mats.size(); i++)
	{
		transpiledSource += ",\n";
		transpiledSource += std::string(mats[i].start, mats[i].length);
	}
	transpiledSource += "\n};\n";

	for (int i = 0; i < sdfs.size(); i++)
	{
		transpiledSource += "__device__ constexpr ";
		switch (sdfs[i].type)
		{
		case SDF::Type::Box:
			transpiledSource += "BoxTest ";
			break;

		case SDF::Type::Sphere:
			transpiledSource += "SphereTest ";
			break;

		case SDF::Type::Transform:
			transpiledSource += "TransformTest ";
			break;
		}
		transpiledSource += std::string(sdfs[i].start, sdfs[i].length);
		transpiledSource += '(';
		for (int j = 0; j < sdfs[i].params.size() - 1; j++)
		{
			std::vector<Token>& v = sdfs[i].params[j];
			if (v.size() > 1)
			{
				transpiledSource += "{ ";
				for (int k = 0; k < v.size() - 1; k++)
				{
					transpiledSource += std::string(v[k].start, v[k].length);
					transpiledSource += ", ";
				}
				transpiledSource += std::string(v[v.size() - 1].start, v[v.size() - 1].length);
				transpiledSource += " }, ";
			}
			else if (v.size() > 0)
			{
				transpiledSource += std::string(v[0].start, v[0].length);
				transpiledSource += ", ";
			}
		}
		std::vector<Token>& v = sdfs[i].params[sdfs[i].params.size() - 1];
		if (v.size() > 1)
		{
			transpiledSource += "{ ";
			for (int k = 0; k < v.size() - 1; k++)
			{
				transpiledSource += std::string(v[k].start, v[k].length);
				transpiledSource += ", ";
			}
			transpiledSource += std::string(v[v.size() - 1].start, v[v.size() - 1].length);
			transpiledSource += " });\n";
		}
		else if (v.size() > 0)
		{
			transpiledSource += std::string(v[0].start, v[0].length);
			transpiledSource += ");\n";
		}
	}

	for (int i = 0; i < mats.size(); i++)
	{
		transpiledSource += "__device__ constexpr auto Query";
		transpiledSource += std::string(mats[i].start, mats[i].length);
		transpiledSource += " = ";
		transpiledSource += std::string(mats[i].paramStart, mats[i].paramLength);
		transpiledSource += ";\n";
	}

	transpiledSource += "__device__ float QueryScene(Vec position, int& hitType)\n{\n";
	transpiledSource += "float distance = 1000000.0f;\nhitType = HT_none;\n";
	for (int i = 0; i < mats.size(); i++)
	{
		std::string matName = std::string(mats[i].start, mats[i].length);
		transpiledSource += "float distance";
		transpiledSource += matName;
		transpiledSource += " = Query";
		transpiledSource += matName;
		transpiledSource += "(position);\n";
		transpiledSource += "if (distance";
		transpiledSource += matName;
		transpiledSource += " < distance)\n{\ndistance = distance";
		transpiledSource += matName;
		transpiledSource += ";\nhitType = HT_";
		transpiledSource += matName;
		transpiledSource += ";\n}\n";
	}
	transpiledSource += "return distance;\n}\n#include \"utils_end.h\"";

	nvrtcProgram mainProgram;
	char* mainPTX;
	size_t mainPTXsize;
	nvrtcProgram noneProgram;
	char* nonePTX;
	size_t nonePTXsize;
	nvrtcProgram* matProgs = new nvrtcProgram[mats.size()];
	char** matPTX = new char*[mats.size()];
	size_t* matPTXsize = new size_t[mats.size()];

	nvrtcCreateProgram(&mainProgram, transpiledSource.c_str(), "QuerySceneUserFn", 0, nullptr, nullptr);
	nvrtcAddNameExpression(mainProgram, "Render");

	char versionOption[20]{ '\0' };
	int major = 5;
	int minor = 2;
	cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice);
	cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice);

	sprintf(versionOption, "-arch=sm_%d%d", major, minor);
	const char* defaultOpts[] = { "-std=c++17", "-I.", "-dc", versionOption, "-DHT_none=0"};
	constexpr size_t NUM_DEFAULT_OPTS = sizeof(defaultOpts) / sizeof(*defaultOpts);
	const char** opts = new const char*[NUM_DEFAULT_OPTS + mats.size()];
	for (int i = 0; i < NUM_DEFAULT_OPTS; i++)
	{
		opts[i] = defaultOpts[i];
	}
	std::vector<std::string> matOpts = std::vector<std::string>(NUM_DEFAULT_OPTS + mats.size());
	for (int i = 0; i < mats.size(); i++)
	{
		matOpts[i] = "-DHT_"s + std::string(mats[i].start, mats[i].length) + '=' + std::to_string(i + 1);
		opts[i + NUM_DEFAULT_OPTS] = matOpts[i].c_str();
	}
	nvrtcResult result = nvrtcCompileProgram(mainProgram, NUM_DEFAULT_OPTS + mats.size(), opts);

	if (result != NVRTC_SUCCESS)
	{
		size_t logSize;
		nvrtcGetProgramLogSize(mainProgram, &logSize);
		char* log = new char[logSize];
		nvrtcGetProgramLog(mainProgram, log);
		printf("Failed to compile generated file:\n%s\n", log);
		nvrtcDestroyProgram(&mainProgram);
		delete[] log;
		delete[] matProgs;
		delete[] matPTX;
		delete[] matPTXsize;
		return false;
	}
	else
	{
		nvrtcGetPTXSize(mainProgram, &mainPTXsize);
		mainPTX = new char[mainPTXsize];
		nvrtcGetPTX(mainProgram, mainPTX);
		nvrtcGetLoweredName(mainProgram, "Render", &loweredName);
	}

	{
		FILE* fp = fopen("none.h", "rb");
		if (fp)
		{
			fseek(fp, 0, SEEK_END);
			size_t len = ftell(fp);
			fseek(fp, 0, SEEK_SET);
			char* buffer = new char[len + 1]{ '\0' };
			fread(buffer, sizeof(char), len, fp);
			fclose(fp);

			nvrtcCreateProgram(&noneProgram, buffer, "none", 0, nullptr, nullptr);
			result = nvrtcCompileProgram(noneProgram, NUM_DEFAULT_OPTS + mats.size(), opts);

			if (result != NVRTC_SUCCESS)
			{
				size_t logSize;
				nvrtcGetProgramLogSize(noneProgram, &logSize);
				char* log = new char[logSize];
				nvrtcGetProgramLog(noneProgram, log);
				printf("Failed to compile material none:\n%s\n", log);
				delete[] log;
				nvrtcDestroyProgram(&mainProgram);
				nvrtcDestroyProgram(&noneProgram);
				delete[] matProgs;
				delete[] matPTX;
				delete[] matPTXsize;
				return false;
			}
			else
			{
				nvrtcGetPTXSize(noneProgram, &nonePTXsize);
				nonePTX = new char[nonePTXsize];
				nvrtcGetPTX(noneProgram, nonePTX);
			}
		}
		else
		{
			nvrtcDestroyProgram(&mainProgram);
			nvrtcDestroyProgram(&noneProgram);
			delete[] matProgs;
			delete[] matPTX;
			delete[] matPTXsize;
			return false;
		}
	}

	for (int i = 0; i < mats.size(); i++)
	{
		FILE* fp = fopen((std::string(mats[i].start, mats[i].length) + ".h").c_str(), "rb");
		if (fp)
		{
			fseek(fp, 0, SEEK_END);
			size_t len = ftell(fp);
			fseek(fp, 0, SEEK_SET);
			char* buffer = new char[len + 1]{ '\0' };
			fread(buffer, sizeof(char), len, fp);
			fclose(fp);

			nvrtcCreateProgram(&matProgs[i], buffer, std::string(mats[i].start, mats[i].length).c_str(), 0, nullptr, nullptr);
			result = nvrtcCompileProgram(matProgs[i], NUM_DEFAULT_OPTS + mats.size(), opts);

			if (result != NVRTC_SUCCESS)
			{
				size_t logSize;
				nvrtcGetProgramLogSize(matProgs[i], &logSize);
				char* log = new char[logSize];
				nvrtcGetProgramLog(matProgs[i], log);
				printf("Failed to compile material file %s:\n%s\n", std::string(mats[i].start, mats[i].length).c_str(), log);
				delete[] log;
				nvrtcDestroyProgram(&mainProgram);
				nvrtcDestroyProgram(&noneProgram);
				for (int j = 0; j < i; j++)
				{
					nvrtcDestroyProgram(&matProgs[j]);
				}
				delete[] matProgs;
				for (int j = 0; j < i; j++)
				{
					delete[] matPTX[j];
				}
				delete[] matPTX;
				delete[] matPTXsize;
				delete[] mainPTX;
				delete[] nonePTX;
				return false;
			}
			else
			{
				nvrtcGetPTXSize(matProgs[i], &matPTXsize[i]);
				matPTX[i] = new char[matPTXsize[i]];
				nvrtcGetPTX(matProgs[i], matPTX[i]);
			}

			delete[] buffer;
		}
		else
		{
			printf("Unable to open material file %s.\n", std::string(mats[i].start, mats[i].length).c_str());
			nvrtcDestroyProgram(&mainProgram);
			nvrtcDestroyProgram(&noneProgram);
			for (int j = 0; j < i; j++)
			{
				nvrtcDestroyProgram(&matProgs[j]);
			}
			delete[] matProgs;
			for (int j = 0; j < i; j++)
			{
				delete[] matPTX[j];
			}
			delete[] matPTX;
			delete[] matPTXsize;
			delete[] mainPTX;
			delete[] nonePTX;
			return false;
		}
	}

	CUlinkState linkState;
	cuLinkCreate(0, nullptr, nullptr, &linkState);

	CUresult res = cuLinkAddData(linkState, CU_JIT_INPUT_PTX, mainPTX, mainPTXsize, "QuerySceneUserFn", 0, nullptr, nullptr);
	res = cuLinkAddData(linkState, CU_JIT_INPUT_PTX, nonePTX, nonePTXsize, "none", 0, nullptr, nullptr);
	for (int i = 0; i < mats.size(); i++)
	{
		res = cuLinkAddData(linkState, CU_JIT_INPUT_PTX, matPTX[i], matPTXsize[i], std::string(mats[i].start, mats[i].length).c_str(), 0, nullptr, nullptr);
	}

	char* image;
	size_t cubinSize;
	res = cuLinkComplete(linkState, (void**)&image, &cubinSize);

	cuModuleLoadData(&cuModule, image);

	cuLinkDestroy(linkState);

	nvrtcDestroyProgram(&mainProgram);
	nvrtcDestroyProgram(&noneProgram);
	for (int i = 0; i < mats.size(); i++)
	{
		nvrtcDestroyProgram(&matProgs[i]);
	}
	delete[] opts;
	delete[] matProgs;
	for (int i = 0; i < mats.size(); i++)
	{
		delete[] matPTX[i];
	}
	delete[] matPTX;
	delete[] matPTXsize;
	delete[] nonePTX;
	delete[] mainPTX;

	return true;
}

void Transpiler::Box()
{
	Consume(Token::Type::Box, "Expect box statement.");
	MakeSDF(SDF::Type::Box);
}

void Transpiler::Sphere()
{
	Consume(Token::Type::Sphere, "Expect sphere statement.");
	MakeSDF(SDF::Type::Sphere);
}

void Transpiler::Transform()
{
	Consume(Token::Type::Transform, "Expect transform statement.");
	MakeSDF(SDF::Type::Transform);
}

void Transpiler::MakeSDF(SDF::Type type)
{
	Consume(Token::Type::Comma, "Expect comma.");
	Consume(Token::Type::String, "Expect identifier.");
	SDF sdf;
	sdf.type = type;
	sdf.start = previous.start;
	sdf.length = previous.length;
	do
	{
		Consume(Token::Type::Param, "Expect param statement (>).");
		sdf.params.emplace_back();
		do
		{
			switch (current.type)
			{
			case Token::Type::Number:
				Consume(Token::Type::Number, "Expect number.");
				break;

			case Token::Type::String:
				Consume(Token::Type::String, "Expect identifier.");
				break;

				ErrorAtCurrent("Expect number or identifier.");
			}
			sdf.params.back().push_back(previous);
			if (current.type == Token::Type::Comma) { Consume(Token::Type::Comma, "Expect comma."); }
		} while (current.type == Token::Type::Number || current.type == Token::Type::String);
	} while (current.type == Token::Type::Param);
	sdfs.push_back(sdf);
}

void Transpiler::MakeMatInternal()
{
	while (current.type == Token::Type::String || current.type == Token::Type::Ampersand || current.type == Token::Type::Pipe || current.type == Token::Type::Minus || current.type == Token::Type::OpenParen || current.type == Token::Type::CloseParen)
	{
		Advance();
		if (previous.type == Token::Type::String)
		{
			if (current.type != Token::Type::OpenParen)
			{
				bool found = false;
				for (int i = 0; i < sdfs.size(); i++)
				{
					if (sdfs[i].length == previous.length && strncmp(sdfs[i].start, previous.start, previous.length) == 0)
					{
						found = true;
						break;
					}
				}
				if (!found) { Error("Undeclared SDF name."); }
			}
			else
			{
				MakeMatInternal();
			}
		}
	};
}

void Transpiler::Advance()
{
	previous = current;
	
	while (true)
	{
		current = lexer.ScanToken();
		if (current.type != Token::Type::Error) { break; }

		ErrorAtCurrent(current.start);
	}
}

void Transpiler::Consume(Token::Type type, const char* message)
{
	if (current.type == type)
	{
		Advance();
		return;
	}

	ErrorAtCurrent(message);
}

void Transpiler::ErrorAtCurrent(const char* errmsg)
{
	ErrorAt(current, errmsg);
}

void Transpiler::Error(const char* errmsg)
{
	ErrorAt(previous, errmsg);
}

void Transpiler::ErrorAt(Token& token, const char* errmsg)
{
	if (panicMode) { return; }

	panicMode = true;

	printf("[Line %d] Error", token.line);

	if (token.type == Token::Type::End) { printf(" at end"); }
	else if (token.type == Token::Type::Error) { }
	else { printf(" at '%.*s'", token.length, token.start); }

	printf(": %s\n", errmsg);
	hadError = true;
}