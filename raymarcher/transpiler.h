#pragma once

#include "lexer.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <vector>

class Transpiler
{
public:
	Transpiler(const char* source);

	bool Transpile(CUdevice& cuDevice, CUmodule& cuModule, const char*& loweredName);

private:
	struct SDF
	{
		int sdfIndex;

		const char* start;
		int length;
		std::vector<std::vector<Token>> params;
	};

	struct Mat
	{
		const char* start;
		int length;

		const char* paramStart;
		int paramLength;
	};

	Lexer lexer;
	Token current;
	Token previous;
	bool hadError;
	bool panicMode;
	std::vector<SDF> sdfs;
	std::vector<Mat> mats;

	void MakeSDF();

	void MakeMatInternal();

	void Advance();
	void Consume(Token::Type type, const char* message);

	void ErrorAtCurrent(const char* errmsg);
	void Error(const char* errmsg);
	void ErrorAt(Token& token, const char* errmsg);
};