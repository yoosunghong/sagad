// Phase 5 scatter tool: place K HISM instances and write each one's variant vector
// as Per-Instance Custom Data, so the shared WPO material renders K distinct
// instances from one mesh file (docs/ARCHITECT.md sec. 2.5 / 3).
//
// Flow: USagadBridge::SampleVariants(asset, K) -> K float rows ->
//   HISM.NumCustomDataFloats = fields.Num();
//   per instance: AddInstance(transform) + SetCustomDataValue(i, c, value).
// The material reads PerInstanceCustomData[c] and combines it with the baked vertex
// colors (bake convention: A = mobility) to deform each instance differently.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "SagadScatterActor.generated.h"

class UHierarchicalInstancedStaticMeshComponent;

UCLASS()
class SAGAD_API ASagadScatterActor : public AActor
{
	GENERATED_BODY()

public:
	ASagadScatterActor();

	/** The baked base mesh (import <asset>_baked.glb so it carries the shared vertex colors). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Sagad|Source")
	TObjectPtr<UStaticMesh> BakedMesh;

	/** The shared WPO material (M_SagadWPO) reading vertex color + Per-Instance Custom Data. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Sagad|Source")
	TObjectPtr<UMaterialInterface> WpoMaterial;

	/** Asset name with a trained diffusion checkpoint (must match the baked mesh). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Sagad|Sampling")
	FString Asset = TEXT("gray-big-rock");

	/** Number of instances to scatter (one sampled variant vector each). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Sagad|Sampling", meta = (ClampMin = "1"))
	int32 Count = 64;

	/** RNG seed: shared by the Python sampler AND the placement grid (reproducible scatter). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Sagad|Sampling")
	int32 Seed = 0;

	/** Half-extent (cm) of the square scatter area on the XY plane. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Sagad|Placement", meta = (ClampMin = "0"))
	float AreaHalfSize = 5000.0f;

	/** Uniform mesh scale applied to every instance (normalized mesh is ~unit radius). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Sagad|Placement", meta = (ClampMin = "0.0001"))
	float InstanceScale = 100.0f;

	/** Optional override for the conda `sagad` python.exe (empty -> SagadBridge default). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Sagad|Bridge")
	FString PythonExe;

	/** Optional override for the sagad repo root (empty -> SagadBridge default). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Sagad|Bridge")
	FString RepoRoot;

	/** Sample variants and (re)build the HISM with Per-Instance Custom Data. */
	UFUNCTION(CallInEditor, BlueprintCallable, Category = "Sagad")
	void Scatter();

	/** Remove all instances. */
	UFUNCTION(CallInEditor, BlueprintCallable, Category = "Sagad")
	void ClearInstances();

private:
	UPROPERTY(VisibleAnywhere, Category = "Sagad")
	TObjectPtr<UHierarchicalInstancedStaticMeshComponent> HISM;
};
