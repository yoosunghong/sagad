// Phase 5 scatter tool implementation. See SagadScatterActor.h.

#include "SagadScatterActor.h"

#include "SagadBridge.h"
#include "Components/HierarchicalInstancedStaticMeshComponent.h"
#include "Engine/StaticMesh.h"
#include "Materials/MaterialInterface.h"
#include "Math/RandomStream.h"

DEFINE_LOG_CATEGORY_STATIC(LogSagadScatter, Log, All);

ASagadScatterActor::ASagadScatterActor()
{
	PrimaryActorTick.bCanEverTick = false;

	HISM = CreateDefaultSubobject<UHierarchicalInstancedStaticMeshComponent>(TEXT("HISM"));
	RootComponent = HISM;
}

void ASagadScatterActor::ClearInstances()
{
	if (HISM)
	{
		HISM->ClearInstances();
		HISM->NumCustomDataFloats = 0;
	}
}

void ASagadScatterActor::Scatter()
{
	if (!HISM)
	{
		return;
	}
	if (!BakedMesh)
	{
		UE_LOG(LogSagadScatter, Error, TEXT("Scatter: BakedMesh is not set."));
		return;
	}

	// -- 1. pull K variant vectors from the Python diffusion sampler ----------
	TArray<FString> Fields;
	TArray<FSagadVariant> Variants;
	FString Error;
	const bool bOk = USagadBridge::SampleVariants(
		PythonExe, RepoRoot, Asset, Count, Seed, Fields, Variants, Error);
	if (!bOk)
	{
		UE_LOG(LogSagadScatter, Error, TEXT("Scatter: sampling failed: %s"), *Error);
		return;
	}
	if (Variants.Num() == 0 || Fields.Num() == 0)
	{
		UE_LOG(LogSagadScatter, Error, TEXT("Scatter: sampler returned no variants."));
		return;
	}

	// -- 2. configure the HISM ------------------------------------------------
	HISM->ClearInstances();
	HISM->SetStaticMesh(BakedMesh);
	if (WpoMaterial)
	{
		HISM->SetMaterial(0, WpoMaterial);
	}
	// One custom-data float per variant field; layout/order matches `Fields`
	// (e.g. [0]=bend_gain, [1]=noise_gain, [2]=scale_gain, [3]=base_band).
	HISM->NumCustomDataFloats = Fields.Num();

	// -- 3. place + write Per-Instance Custom Data ----------------------------
	// Seed shared with the Python sampler so the whole scatter is reproducible.
	FRandomStream Rng(Seed);
	const int32 GridN = FMath::Max(1, FMath::CeilToInt(FMath::Sqrt((float)Variants.Num())));
	const float Step = (GridN > 1) ? (2.0f * AreaHalfSize / (GridN - 1)) : 0.0f;
	const float Jitter = Step * 0.35f;

	for (int32 i = 0; i < Variants.Num(); ++i)
	{
		const int32 Gx = i % GridN;
		const int32 Gy = i / GridN;
		const float X = -AreaHalfSize + Gx * Step + Rng.FRandRange(-Jitter, Jitter);
		const float Y = -AreaHalfSize + Gy * Step + Rng.FRandRange(-Jitter, Jitter);

		FTransform Xf;
		Xf.SetLocation(FVector(X, Y, 0.0f));
		Xf.SetRotation(FRotator(0.0f, Rng.FRandRange(0.0f, 360.0f), 0.0f).Quaternion());
		Xf.SetScale3D(FVector(InstanceScale));

		const int32 Idx = HISM->AddInstance(Xf);  // local space (root-relative)

		const FSagadVariant& V = Variants[i];
		const int32 NumData = FMath::Min(V.Values.Num(), Fields.Num());
		for (int32 c = 0; c < NumData; ++c)
		{
			HISM->SetCustomDataValue(Idx, c, V.Values[c], /*bMarkRenderStateDirty=*/false);
		}
	}

	HISM->MarkRenderStateDirty();
	UE_LOG(LogSagadScatter, Log,
		TEXT("Scatter: placed %d instances, %d custom-data floats each (asset=%s seed=%d)."),
		Variants.Num(), Fields.Num(), *Asset, Seed);
}
