#include "utils.h"

__device__ bool sun(Vec& origin, Vec& direction, Vec& color, Vec& attenuation, Vec sampledPosition, Vec normal, xorwow& random)
{
	color += attenuation * Vec(50.0f, 42.5f, 32.5f);
	return false;
}