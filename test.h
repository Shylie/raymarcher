#include "utils.h"

__device__ constexpr BoxTest lowerRoom(Vec(-60.0f, -0.5f, -30.0f), Vec(35.0f, 100.0f, 60.0f));
__device__ constexpr BoxTest upperRoom(Vec(-50.0f, 16.0f, -25.0f), Vec(25.0f, 20.0f, 50.0f));
__device__ constexpr BoxTest roomHole1(Vec(29.0f, 7.0f, -15.0f), Vec(35.0f, 15.0f, -3.0f));
__device__ constexpr BoxTest roomHole2(Vec(31.0f, 7.0f, -15.0f), Vec(35.0f, 20.0f, -3.0f));
__device__ constexpr BoxTest roomHole3(Vec(29.0f, 5.0f, -15.0f), Vec(31.0f, 7.5f, -3.0f));
__device__ constexpr BoxTest planks(Vec(0.65f, 18.2f, 0.65f), Vec(3.35f, 19.0f, 3.35f));
__device__ constexpr auto room = -(lowerRoom | upperRoom | roomHole1 | roomHole2 | roomHole3);

__device__ constexpr SphereTest metal1(Vec(-13.0f, 8.0f, 5.0f), 3.0f);
__device__ constexpr SphereTest metal2(Vec(32.0f, 10.5f, -9.0f), 1.8f);
__device__ constexpr SphereTest metal2Cutout(Vec(31.9f, 10.6f, -7.8f), 1.55f);
__device__ constexpr BoxTest metalInGlass(Vec(-6.1f, 5.65f, 4.5f), Vec(-5.5f, 6.65f, 5.5f));
__device__ constexpr auto metal = metal1 | metalInGlass | (metal2 - metal2Cutout);

__device__ constexpr SphereTest glass1(Vec(-6.0f, 6.15f, 5.0f), 1.6f);
__device__ constexpr BoxTest glass1Cutout(Vec(-6.2f, 5.55f, 4.4f), Vec(-5.4f, 6.75f, 5.6f));
__device__ constexpr SphereTest glass2(Vec(6.0f, 2.55f, 6.2f), 1.5f);
__device__ constexpr BoxTest glass3(Vec(-28.0f, 2.0f, 10.0f), Vec(-23.0f, 7.0f, 15.0f));
__device__ constexpr SphereTest glass3Cutout((Vec(-28.0f, 2.0f, 10.0f) + Vec(-23.0f, 7.0f, 15.0f)) / 2, 2.4f);
__device__ constexpr auto glass = (glass1 - glass1Cutout) | glass2 | (glass3 - glass3Cutout);

__device__ constexpr SphereTest sunBall((Vec(-28.0f, 2.0f, 10.0f) + Vec(-23.0f, 7.0f, 15.0f)) / 2, 2.0f);
__device__ constexpr BoxTest sunBox(Vec(-15.0f, 35.0f, -15.0f), Vec(15.0f, 36.0f, 15.0f));
__device__ constexpr auto sun = sunBall | sunBox;

__device__ float QueryScene(Vec position, HitType& hitType)
{
	float distance = metal(position);
	hitType = HT_METAL;

	float glassDistance = glass(position);
	if (glassDistance < distance)
	{
		distance = glassDistance;
		hitType = HT_GLASS;
	}

	float roomDistance = Min(room(position), planks(Vec(fmod(fabs(position.X()), 4.0f), position.Y(), fmod(fabs(position.Z()), 4.0f))));
	if (roomDistance < distance)
	{
		distance = roomDistance;
		hitType = HT_WALL;
	}

	float sunDistance = sun(position);
	if (sunDistance < distance)
	{
		distance = sunDistance;
		hitType = HT_SUN;
	}

	return distance;
}

#include "utils_end.h"