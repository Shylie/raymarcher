#include "scanner.h"

#include <cstring>

Scanner::Scanner(const char* source) : start(source), current(source), line(1) {}

Token Scanner::Scan()
{
	start = current;

	if (IsAtEnd()) { return MakeToken(Token::Type::End); }

	char c = Advance();

	switch (c)
	{
	}

	return ErrorToken("Unexpected character.");
}

char Scanner::Advance()
{
	current++;
	return current[-1];
}

bool Scanner::IsAtEnd() const
{
	return *current == '\0';
}

Token Scanner::MakeToken(Token::Type type) const
{
	Token token;
	token.type = type;
	token.start = start;
	token.length = (int)(current - start);
	token.line = line;
	return token;
}

Token Scanner::ErrorToken(const char* message) const
{
	Token token;
	token.type = Token::Type::Error;
	token.start = message;
	token.length = (int)strlen(message);
	token.line = line;
	return token;
}