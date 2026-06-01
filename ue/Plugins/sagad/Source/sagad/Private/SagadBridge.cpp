// Phase 5 subprocess bridge implementation. See SagadBridge.h.

#include "SagadBridge.h"

#include "HAL/PlatformProcess.h"
#include "Misc/Paths.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"

DEFINE_LOG_CATEGORY_STATIC(LogSagadBridge, Log, All);

// These mirror the environment recorded in the sagad repo MEMORY (the bare `python`
// on PATH is a broken Store stub -- always call the env interpreter by full path).
FString USagadBridge::DefaultPythonExe()
{
	return TEXT("C:/Users/PC/anaconda3/envs/sagad/python.exe");
}

FString USagadBridge::DefaultRepoRoot()
{
	// The UE project lives at <repo>/ue/, so the sagad Python repo root (which holds
	// scripts/sample_variants.py) is the project directory's parent. Derived rather
	// than hardcoded so the bridge survives a clone to a different path.
	return FPaths::ConvertRelativePathToFull(FPaths::Combine(FPaths::ProjectDir(), TEXT("..")));
}

bool USagadBridge::SampleVariants(
	const FString& PythonExe,
	const FString& RepoRoot,
	const FString& Asset,
	int32 K,
	int32 Seed,
	TArray<FString>& OutFields,
	TArray<FSagadVariant>& OutVariants,
	FString& OutError)
{
	OutFields.Reset();
	OutVariants.Reset();
	OutError.Reset();

	const FString Py = PythonExe.IsEmpty() ? DefaultPythonExe() : PythonExe;
	const FString Root = RepoRoot.IsEmpty() ? DefaultRepoRoot() : RepoRoot;
	const FString Script = FPaths::Combine(Root, TEXT("scripts"), TEXT("sample_variants.py"));

	if (!FPaths::FileExists(Py))
	{
		OutError = FString::Printf(TEXT("python.exe not found: %s"), *Py);
		UE_LOG(LogSagadBridge, Error, TEXT("%s"), *OutError);
		return false;
	}
	if (!FPaths::FileExists(Script))
	{
		OutError = FString::Printf(TEXT("sampler script not found: %s"), *Script);
		UE_LOG(LogSagadBridge, Error, TEXT("%s"), *OutError);
		return false;
	}

	// Quote the script path (may contain spaces); flags match scripts/sample_variants.py.
	const FString Params = FString::Printf(
		TEXT("\"%s\" --asset %s --k %d --seed %d"), *Script, *Asset, K, Seed);

	int32 ReturnCode = -1;
	FString StdOut;
	FString StdErr;

	UE_LOG(LogSagadBridge, Log, TEXT("sampling variants: %s %s"), *Py, *Params);
	const bool bLaunched = FPlatformProcess::ExecProcess(
		*Py, *Params, &ReturnCode, &StdOut, &StdErr, *Root);

	if (!bLaunched)
	{
		OutError = FString::Printf(TEXT("failed to launch python: %s"), *Py);
		UE_LOG(LogSagadBridge, Error, TEXT("%s"), *OutError);
		return false;
	}

	// stdout is contract-bound to a single JSON object (logging is on stderr).
	TSharedPtr<FJsonObject> Json;
	const TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(StdOut);
	if (!FJsonSerializer::Deserialize(Reader, Json) || !Json.IsValid())
	{
		OutError = FString::Printf(
			TEXT("unparseable sampler output (exit=%d). stdout=%s stderr=%s"),
			ReturnCode, *StdOut.Left(512), *StdErr.Left(512));
		UE_LOG(LogSagadBridge, Error, TEXT("%s"), *OutError);
		return false;
	}

	// The Python side reports failures as {"error": "..."} with a non-zero exit.
	FString PyError;
	if (Json->TryGetStringField(TEXT("error"), PyError))
	{
		OutError = FString::Printf(TEXT("sampler error: %s"), *PyError);
		UE_LOG(LogSagadBridge, Error, TEXT("%s"), *OutError);
		return false;
	}

	const TArray<TSharedPtr<FJsonValue>>* FieldsJson = nullptr;
	const TArray<TSharedPtr<FJsonValue>>* VariantsJson = nullptr;
	if (!Json->TryGetArrayField(TEXT("fields"), FieldsJson) ||
		!Json->TryGetArrayField(TEXT("variants"), VariantsJson))
	{
		OutError = TEXT("sampler JSON missing 'fields' or 'variants'");
		UE_LOG(LogSagadBridge, Error, TEXT("%s"), *OutError);
		return false;
	}

	for (const TSharedPtr<FJsonValue>& F : *FieldsJson)
	{
		OutFields.Add(F->AsString());
	}

	OutVariants.Reserve(VariantsJson->Num());
	for (const TSharedPtr<FJsonValue>& RowVal : *VariantsJson)
	{
		const TArray<TSharedPtr<FJsonValue>>* Row = nullptr;
		if (!RowVal->TryGetArray(Row))
		{
			OutError = TEXT("malformed variant row (expected array of floats)");
			UE_LOG(LogSagadBridge, Error, TEXT("%s"), *OutError);
			return false;
		}
		FSagadVariant Variant;
		Variant.Values.Reserve(Row->Num());
		for (const TSharedPtr<FJsonValue>& Num : *Row)
		{
			Variant.Values.Add(static_cast<float>(Num->AsNumber()));
		}
		OutVariants.Add(MoveTemp(Variant));
	}

	UE_LOG(LogSagadBridge, Log, TEXT("sampled %d variants x %d fields (exit=%d)"),
		OutVariants.Num(), OutFields.Num(), ReturnCode);
	return ReturnCode == 0;
}
