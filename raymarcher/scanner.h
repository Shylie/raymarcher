#pragma once

struct Token
{
	enum class Type
	{
		End,
		Error
	} type;
	const char* start;
	int length;
	int line;
};

class Scanner
{
public:
	Scanner(const char* source);
	Token Scan();

private:
	const char* start;
	const char* current;
	int line;

	char Advance();

	bool IsAtEnd() const;

	Token MakeToken(Token::Type type) const;
	Token ErrorToken(const char* message) const;
};