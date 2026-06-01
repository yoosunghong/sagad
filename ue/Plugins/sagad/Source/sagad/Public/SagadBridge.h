// Phase 5 subprocess bridge: calls the Python variant sampler and parses its JSON.
//
// The Python side (sagad repo: scripts/sample_variants.py) prints ONE JSON object to
// stdout and routes all logging to stderr, so the contract here is: run the conda
// `sagad` python on that script, read stdout, deserialize. Per-instance variant
// vectors come back as float rows aligned to `OutFields` -- exactly the order to push
// into HISM/Nanite Per-Instance Custom Data (docs/ARCHITECT.md sec. 2.5 / 3).

#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "SagadBridge.generated.h"

/** One sampled per-instance variant: float values aligned to USagadBridge fields. */
USTRUCT(BlueprintType)
struct FSagadVariant
{
	GENERATED_BODY()

	/** Raw Per-Instance Custom Data payload, ordered to match the field list. */
	UPROPERTY(BlueprintReadOnly, Category = "Sagad")
	TArray<float> Values;
};

UCLASS()
class SAGAD_API USagadBridge : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()

public:
	/**
	 * Sample K per-instance variant vectors by invoking the Python diffusion sampler.
	 *
	 * @param PythonExe  Full path to the `sagad` conda python.exe. Empty -> built-in default.
	 * @param RepoRoot   Path to the sagad repo root (holds scripts/). Empty -> built-in default.
	 * @param Asset      Asset name with a trained checkpoint (e.g. "gray-big-rock").
	 * @param K          Number of instances / variant vectors to sample.
	 * @param Seed       RNG seed for reproducible scatter.
	 * @param OutFields  Field names in PICD write order (e.g. bend, noise, scale, base_band).
	 * @param OutVariants K rows, each Values[] aligned to OutFields.
	 * @param OutError   Human-readable failure reason when the call returns false.
	 * @return true on success (valid JSON, no error field, exit 0).
	 */
	UFUNCTION(BlueprintCallable, Category = "Sagad",
		meta = (AdvancedDisplay = "PythonExe,RepoRoot"))
	static bool SampleVariants(
		const FString& PythonExe,
		const FString& RepoRoot,
		const FString& Asset,
		int32 K,
		int32 Seed,
		TArray<FString>& OutFields,
		TArray<FSagadVariant>& OutVariants,
		FString& OutError);

	/** Built-in default conda `sagad` python.exe path (overridable per call). */
	static FString DefaultPythonExe();

	/** Built-in default sagad repo root (overridable per call). */
	static FString DefaultRepoRoot();
};
