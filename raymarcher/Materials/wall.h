#include "utils.h"

__device__ bool wall(Vec origin, Vec& direction, Vec& color, Vec& attenuation, Vec sampledPosition, Vec normal, xorwow& random)
{
	float p = 6.283185f * random();
	float c = random();
	float s = sqrt(1 - c);
	float g = normal.Z() < 0.0f ? -1.0f : 1.0f;
	float u = -1.0f / (g + normal.Z());
	float v = normal.X() * normal.Y() * u;
	direction = Vec(v, g + normal.Y() * normal.Y() * u, -normal.Y()) * (cos(p) * s) + Vec(1 + g * normal.X() * normal.X() * u, g * v, -g * normal.X()) * (sin(p) * s) + normal * sqrt(c);
	origin = sampledPosition + direction * 0.1f;
	attenuation *= 0.25f;

	return true;
}