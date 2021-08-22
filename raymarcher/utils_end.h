__device__ HitType March(Vec origin, Vec direction, Vec& hitPos, Vec& hitNorm)
{
	constexpr float TEST_DISTANCE = 0.01f;

	HitType hitType = HT_NONE;
	int noHitCount = 0;
	float distance = 0;

	for (float totalDistance = 0.0f; totalDistance < 1000.0f && noHitCount < 100; totalDistance += distance)
	{
		distance = QueryScene(hitPos = origin + direction * totalDistance, hitType);

		while (distance < -TEST_DISTANCE && ++noHitCount < 99)
		{
			totalDistance += fabs(distance);
			distance = QueryScene(hitPos = origin + direction * totalDistance, hitType);
		}

		if (fabs(distance) < TEST_DISTANCE || ++noHitCount > 99)
		{
			HitType ignored;
			hitNorm = !Vec(QueryScene(hitPos + Vec(TEST_DISTANCE, 0.0f, 0.0f), ignored) - distance, QueryScene(hitPos + Vec(0.0f, TEST_DISTANCE, 0.0f), ignored) - distance, QueryScene(hitPos + Vec(0.0f, 0.0f, TEST_DISTANCE), ignored) - distance);
			if (distance < 0.0f)
			{
				hitNorm = -hitNorm;
			}

			return hitType;
		}
	}

	return HT_NONE;
}

__device__ Vec Trace(Vec origin, Vec direction, xorwow& random)
{
	static constexpr Vec LIGHT_DIRECTION(0.45749f, 0.46749f, 0.36249f);
	static const int BOUNCE_COUNT = 10;
	Vec sampledPosition, normal, color, attenuation = 1;

	for (int bc = 0; bc < BOUNCE_COUNT; bc++)
	{
		HitType hitType = March(origin, direction, sampledPosition, normal);
		if (hitType == HT_NONE) { break; }
		else if (hitType == HT_GLASS)
		{
			float REFRACTION_INDEX = 1.54f;
			float R0 = (1.0f - REFRACTION_INDEX) / (1.0f + REFRACTION_INDEX);
			float R1 = R0 * R0;

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
		}
		else if (hitType == HT_METAL)
		{
			direction += normal * (-2 * (normal % direction));
			origin = sampledPosition + direction * 0.1f;
			attenuation *= 0.3f;
		}
		else if (hitType == HT_WALL)
		{
			float incidence = normal % LIGHT_DIRECTION;
			float p = 6.283185f * random();
			float c = random();
			float s = sqrt(1 - c);
			float g = normal.Z() < 0.0f ? -1.0f : 1.0f;
			float u = -1.0f / (g + normal.Z());
			float v = normal.X() * normal.Y() * u;
			direction = Vec(v, g + normal.Y() * normal.Y() * u, -normal.Y()) * (cos(p) * s) + Vec(1 + g * normal.X() * normal.X() * u, g * v, -g * normal.X()) * (sin(p) * s) + normal * sqrt(c);
			origin = sampledPosition + direction * 0.1f;
			attenuation *= 0.2f;
			if (incidence > 0.0f && March(sampledPosition + normal * 0.1f, LIGHT_DIRECTION, sampledPosition, normal) == HT_SUN)
			{
				color += attenuation * Vec(50.0f, 42.5f, 32.5f) * incidence * 1.2;
			}
		}
		else if (hitType == HT_SUN)
		{
			color += attenuation * Vec(50.0f, 42.5f, 32.5f);
			break;
		}
	}

	return color;
}

__global__ void Render(Vec* cols, int w, int h, int s, Vec position, Vec goal, Vec left, Vec up)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < w; i += blockDim.x * gridDim.x)
	{
		for (int j = threadIdx.y + blockIdx.y * blockDim.y; j < h; j += blockDim.y * gridDim.y)
		{
			xorwow random;
			random.a = i + 2;
			random.b = j + 2;

			Vec col;
			for (int sample = 0; sample < s; sample++)
			{
				col += Trace(position, !(goal + left * (i - w / 2.0f + random()) + up * (j - h / 2.0f + random())), random);
			}
			cols[i + j * w] = col / s;
		}
	}
}