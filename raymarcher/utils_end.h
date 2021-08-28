__device__ int March(Vec origin, Vec direction, Vec& hitPos, Vec& hitNorm)
{
	constexpr float TEST_DISTANCE = 0.0001f;

	int hitType = HT_none;
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
			int ignored;
			hitNorm = !Vec(QueryScene(hitPos + Vec(TEST_DISTANCE, 0.0f, 0.0f), ignored) - distance, QueryScene(hitPos + Vec(0.0f, TEST_DISTANCE, 0.0f), ignored) - distance, QueryScene(hitPos + Vec(0.0f, 0.0f, TEST_DISTANCE), ignored) - distance);
			if (distance < 0.0f)
			{
				hitNorm = -hitNorm;
			}

			return hitType;
		}
	}

	return HT_none;
}

__device__ Vec Trace(Vec origin, Vec direction, xorwow& random)
{
	static const int BOUNCE_COUNT = 6;
	Vec sampledPosition, normal, color, attenuation = 1;

	for (int bc = 0; bc < BOUNCE_COUNT; bc++)
	{
		int hitType = March(origin, direction, sampledPosition, normal);
		if (!materials[hitType](origin, direction, color, attenuation, sampledPosition, normal, random))
		{
			break;
		}
	}

	return color;
}

__global__ void Render(Vec* cols, int w, int h, int s, Vec position, Vec goal, Vec left, Vec up, int sx, int sy, int ex, int ey)
{
	for (int i = sx + threadIdx.x + blockIdx.x * blockDim.x; i < ex && i < w; i += blockDim.x * gridDim.x)
	{
		for (int j = sy + threadIdx.y + blockIdx.y * blockDim.y; j < ey && j < h; j += blockDim.y * gridDim.y)
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