#pragma once

#include "utils.h"

/*
* EXAMPLE FILE
* ------------
* decmat,wall,metal,glass,sun
*
* sdf box,alpha
* > 1,2,3
* > 3,4,5
*
* sdf sph,beta
* > 4,4,4
* > 2
*
* sdf box,theta
* > -10,-10,-10
* > 10,10,10
*
* mat,wall
* > alpha & beta
*
* mat,metal
* > alpha | beta
*
* mat,glass
* > beta - alpha
*
* mat,sun
* > -theta
*/

struct Token
{
	enum class Type
	{
		Number, // floating-point
		String, // text. could be a string or keyword.
		Comma,
		Param, // greater than symbol.
		Ampersand,
		Pipe,
		Minus,
		OpenParen,
		CloseParen,
		Brace,
		End, // end of source,
		DeclareMaterials,
		Material,
		Box,
		Sphere,
		Transform,
		SDF,
		Error
	} type;

	const char* start; // start of text, could be irrelevant
	int length; // length of string pointed to by this->start
	int line; // line this token was parsed on
};

constexpr const char* DECLARE_MATERIALS = "decmat"; // declare material types
constexpr const char* MATERIAL = "mat"; // declare mat shapes referencing box & sphere tests
constexpr const char* BOX = "box"; // declare a box test
constexpr const char* SPHERE = "sph"; // declare a sphere test
constexpr const char* TRANSFORM = "trf"; // declare a transformation
constexpr const char* SDF = "sdf"; // declare an sdf

class Lexer
{
public:
	using MatchFn = bool (*)(char);

	Lexer(const char* source);

	Token ScanToken();

private:
	const char* start;
	const char* current;
	int line;

	char Advance();
	char Peek() const;
	char PeekNext() const;
	bool Match(MatchFn matcher);
	bool IsAtEnd() const;

	void SkipWhitespace();

	Token MakeToken(Token::Type type) const;
	Token MakeErrorToken(const char* errmsg) const;
};