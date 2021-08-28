#include "lexer.h"

#include <cstdio>
#include <cstring>

static bool IsAlpha(char c)
{
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

static bool IsNum(char c)
{
	return c >= '0' && c <= '9';
}

Lexer::Lexer(const char* source) : start(source), current(source), line(1) {}

Token Lexer::ScanToken()
{
	SkipWhitespace();
	start = current;

	if (IsAtEnd()) { return MakeToken(Token::Type::End); }

	char c = Advance();

	switch (c)
	{
	case '\0': return MakeToken(Token::Type::End);
	case ',': return MakeToken(Token::Type::Comma);
	case '>': return MakeToken(Token::Type::Param);
	case '&': return MakeToken(Token::Type::Ampersand);
	case '|': return MakeToken(Token::Type::Pipe);
	case '(': return MakeToken(Token::Type::OpenParen);
	case ')': return MakeToken(Token::Type::CloseParen);
	case '{':
	case '}':
		return MakeToken(Token::Type::Brace);
	case '-':
		if (Match(IsNum))
		{
			while (Match(IsNum)) {}
			Match([](char c) { return c == '.'; });
			while (Match(IsNum)) {}
			return MakeToken(Token::Type::Number);
		}
		else
		{
			return MakeToken(Token::Type::Minus);
		}

	default:
		if (IsAlpha(c))
		{
			while (Match(IsAlpha)) {}
			return MakeToken(Token::Type::String);
		}
		else if (IsNum(c))
		{
			while (Match(IsNum)) {}
			Match([](char c) { return c == '.'; });
			while (Match(IsNum)); {}
			return MakeToken(Token::Type::Number);
		}
		return MakeErrorToken("Unknown character.");
	}
}

char Lexer::Advance() { return *current++; }
char Lexer::Peek() const { return *current; }
char Lexer::PeekNext() const
{
	if (IsAtEnd()) { return '\0'; }
	return current[1];
}
bool Lexer::Match(MatchFn matcher)
{
	if (IsAtEnd()) { return false; }
	if (!matcher(*current)) { return false; }
	current++;
	return true;
}
bool Lexer::IsAtEnd() const { return *current == '\0'; }

void Lexer::SkipWhitespace()
{
	while (true)
	{
		char c = Peek();
		switch (c)
		{
		case ' ':
		case '\r':
		case '\t':
			Advance();
			break;

		case '\n':
			line++;
			Advance();
			break;

		case '/':
			if (PeekNext() == '/')
			{
				while (Peek() != '\n' && !IsAtEnd()) { Advance(); }
				break;
			}
			else
			{
				return;
			}

		default:
			return;
		}
	}
}

Token Lexer::MakeToken(Token::Type type) const
{
	Token token;
	token.type = type;
	token.start = start;
	token.length = (int)(current - start);
	token.line = line;

	// check for keywords
	if (token.type == Token::Type::String)
	{
		if (strncmp(token.start, DECLARE_MATERIALS, token.length) == 0) { token.type = Token::Type::DeclareMaterials; }
		else if (strncmp(token.start, MATERIAL, token.length) == 0) { token.type = Token::Type::Material; }
		else if (strncmp(token.start, BOX, token.length) == 0) { token.type = Token::Type::Box; }
		else if (strncmp(token.start, SPHERE, token.length) == 0) { token.type = Token::Type::Sphere; }
		else if (strncmp(token.start, TRANSFORM, token.length) == 0) { token.type = Token::Type::Transform; }
		else if (strncmp(token.start, SDF, token.length) == 0) { token.type = Token::Type::SDF; }
	}

	return token;
}
Token Lexer::MakeErrorToken(const char* errmsg) const
{
	Token token;
	token.type = Token::Type::Error;
	token.start = errmsg;
	token.length = (int)strlen(errmsg);
	token.line = line;
	return token;
}