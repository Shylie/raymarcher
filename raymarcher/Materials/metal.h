#include "utils.h"

__device__ bool metal(Vec origin, Vec& direction, Vec& color, Vec& attenuation, Vec sampledPosition, Vec normal, xorwow& random)
{
	direction += normal * (-2 * (normal % direction));
	attenuation *= 0.35f;

	return true;
}