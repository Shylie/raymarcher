#pragma once

#ifndef __device__
#define __device__
#endif

#ifndef __CUDA_ARCH__
#include <cmath>
#include <cstdio>
#endif

__device__ inline float Min(float a, float b) { return a < b ? a : b; }
__device__ inline float Max(float a, float b) { return a > b ? a : b; }
constexpr float PI = 3.1415926535897932384626433832795028841971693993751f;
__device__ constexpr float RAD2DEG(float angle) { return 180.0f / PI * angle; }
__device__ constexpr float DEG2RAD(float angle) { return PI / 180.0f * angle; }

struct xorwow
{
	unsigned int a, b, c, d, e, counter;

	__device__ xorwow() : a(1), b(1), c(1), d(1), e(0), counter(0) { }

	__device__ float operator()()
	{
		unsigned int t = e;
		unsigned int s = a;
		e = d;
		d = c;
		c = b;
		b = s;
		t ^= t >> 2;
		t ^= t << 1;
		t ^= s ^ (s << 4);
		a = t;
		counter += 362437;
		return static_cast<float>(t + counter) / static_cast<float>(0xffffffff);
	}
};

struct Vec
{
	float c[3];

	__device__ constexpr Vec() : Vec(0.0f) { }
	__device__ constexpr Vec(float c) : Vec(c, c, c) { }
	__device__ constexpr Vec(float x, float y, float z) : c{ x, y, z } { }

	__device__ constexpr float X() const { return c[0]; }
	__device__ constexpr float& X() { return c[0]; }
	__device__ constexpr float Y() const { return c[1]; }
	__device__ constexpr float& Y() { return c[1]; }
	__device__ constexpr float Z() const { return c[2]; }
	__device__ constexpr float& Z() { return c[2]; }

	__device__ constexpr float operator[](int index) const { return c[index % 3]; }
	__device__ constexpr float& operator[](int index) { return c[index % 3]; }

#define VEC_OP_EQ(op) __device__ constexpr Vec& operator##op##=(Vec v) { X() op##= v.X(); Y() op##= v.Y(); Z() op##= v.Z(); return *this; }

	VEC_OP_EQ(+);
	VEC_OP_EQ(-);
	VEC_OP_EQ(*);
	VEC_OP_EQ(/);

#undef VEC_OP_EQ
#define STRINGIFY2(x) #x
#define STRINGIFY(x) STRINGIFY2(x)
#define VEC_OP(op) __device__ constexpr Vec operator##op##(Vec v) const { Vec tmp = *this; tmp op##= v; return tmp; }

	VEC_OP(+);
	VEC_OP(-);
	VEC_OP(*);
	VEC_OP(/);

#undef VEC_OP

	__device__ constexpr float operator%(Vec v) const { return X() * v.X() + Y() * v.Y() + Z() * v.Z(); }
	__device__ constexpr Vec operator^(Vec v) const { return Vec(Y() * v.Z() - Z() * v.Y(), Z() * v.X() - X() * v.Z(), X() * v.Y() - Y() * v.X()); }

	__device__ constexpr Vec operator-() const { return *this * -1; }
	__device__ Vec operator!() const { return *this / sqrtf(*this % *this); }
};

template <typename T, typename U>
struct IntersectionTest
{
	T a;
	U b;

	__device__ explicit constexpr IntersectionTest(T a, U b) : a(a), b(b) { }

	__device__ float operator()(Vec position) const
	{
		return Max(a(position), b(position));
	}
};

template <typename T, typename U>
__device__ constexpr auto operator&(T a, U b)
{
	return IntersectionTest<T, U>(a, b);
}

template <typename T, typename U>
struct UnionTest
{
	T a;
	U b;

	__device__ explicit constexpr UnionTest(T a, U b) : a(a), b(b) { }

	__device__ float operator()(Vec position) const
	{
		return Min(a(position), b(position));
	}
};

template <typename T, typename U>
__device__ constexpr auto operator|(T a, U b)
{
	return UnionTest<T, U>(a, b);
}

template <typename T, typename U>
struct DifferenceTest
{
	T a;
	U b;

	__device__ explicit constexpr DifferenceTest(T a, U b) : a(a), b(b) { }

	__device__ float operator()(Vec position) const
	{
		return Max(a(position), -b(position));
	}
};

template <typename T, typename U>
__device__ constexpr auto operator-(T a, U b)
{
	return DifferenceTest<T, U>(a, b);
}

template <typename T>
struct CarveTest
{
	T a;

	__device__ explicit constexpr CarveTest(T a) : a(a) { }

	__device__ float operator()(Vec position) const
	{
		return -a(position);
	}
};

template <typename T>
__device__ constexpr auto operator-(T a)
{
	return CarveTest<T>(a);
}

template <typename T>
struct TransformTest
{
	T a;
	Vec axis;
	float angle;
	Vec translation;

	__device__ explicit constexpr TransformTest(T a, Vec translation) : a(a), axis(), angle(0), translation(translation) {}
	__device__ explicit constexpr TransformTest(T a, Vec axis, float angle) : a(a), axis(axis), angle(DEG2RAD(-angle)), translation() {}
	__device__ explicit constexpr TransformTest(T a, Vec axis, float angle, Vec translation) : a(a), axis(axis), angle(DEG2RAD(-angle)), translation(translation) {}

	__device__ float operator()(Vec position) const
	{
		const float cosAngle = cosf(angle);
		position -= translation;
		Vec transformed = position * cosAngle + (!axis ^ position) * sinf(angle) + !axis * ((1 - cosAngle) * (!axis % position));
		return a(transformed);
	}
};

struct BoxTest
{
	Vec p, s;

	__device__ constexpr BoxTest(Vec sz) : BoxTest(-sz, sz) {}

	__device__ constexpr BoxTest(Vec lower, Vec upper) : p((upper + lower) / 2), s((upper - lower) / 2)
	{
		for (int i = 0; i < 3; i++)
		{
			if (s[i] < 0.0f) { s[i] = -s[i]; }
		}
	}

	__device__ float operator()(Vec position) const
	{
		position -= p;
		for (int i = 0; i < 3; i++)
		{
			position[i] = fabsf(position[i]);
		}
		position -= s;
		
		Vec tmp(Max(position.X(), 0), Max(position.Y(), 0), Max(position.Z(), 0));
		return sqrtf(tmp % tmp) + Min(Max(position.X(), Max(position.Y(), position.Z())), 0);
	}
};

struct SphereTest
{
	Vec center;
	float radius;

	__device__ constexpr SphereTest(float radius) : SphereTest(0, radius) {}
	__device__ constexpr SphereTest(Vec center, float radius) : center(center), radius(radius) { }

	__device__ float operator()(Vec position) const
	{
		Vec delta = center - position;
		float distance = sqrtf(delta % delta);
		return distance - radius;
	}
};

template <typename T>
struct RoundTest
{
	T a;
	float r;

	__device__ explicit constexpr RoundTest(T a, float r) : a(a), r(r) {}

	__device__ float operator()(Vec position) const
	{
		return a(position) - r;
	}
};

__device__ extern int March(Vec origin, Vec direction, Vec& hitPos, Vec& hitNorm);