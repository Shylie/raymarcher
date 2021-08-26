#include "utils.h"

__device__ bool glass(Vec& origin, Vec& direction, Vec& color, Vec& attenuation, Vec sampledPosition, Vec normal, xorwow& random)
{
	static constexpr float REFRACTION_INDEX = 1.54f;
	static constexpr float R0 = (1.0f - REFRACTION_INDEX) / (1.0f + REFRACTION_INDEX);
	static constexpr float R1 = R0 * R0;

	float refractionRatio = direction % normal < 0 ? 1.0f / REFRACTION_INDEX : REFRACTION_INDEX;

	float cosTheta = fmin(-direction % normal, 1.0f);
	float sinTheta = sqrt(1.0f - cosTheta * cosTheta);

	bool cannotRefract = refractionRatio * sinTheta > 1.0f || (R1 + (1.0f - R1) * powf((1 - cosTheta), 5)) > random();

	if (cannotRefract)
	{
		direction += normal * (-2 * (normal % direction));
	}
	else
	{
		Vec rOutPerp = (direction + normal * cosTheta) * refractionRatio;
		Vec rOutParallel = normal * -sqrt(fabs(1.0f - (rOutPerp % rOutPerp)));
		direction = rOutPerp + rOutParallel;
	}

	origin = sampledPosition + direction * 0.1f;

	return true;
}