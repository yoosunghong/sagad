// Copyright Epic Games, Inc. All Rights Reserved.

using UnrealBuildTool;

public class sagad_ue : ModuleRules
{
	public sagad_ue(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[] {
			"Core",
			"CoreUObject",
			"Engine",
			"InputCore",
			"EnhancedInput",
			"AIModule",
			"StateTreeModule",
			"GameplayStateTreeModule",
			"UMG",
			"Slate"
		});

		PrivateDependencyModuleNames.AddRange(new string[] { });

		PublicIncludePaths.AddRange(new string[] {
			"sagad_ue",
			"sagad_ue/Variant_Platforming",
			"sagad_ue/Variant_Platforming/Animation",
			"sagad_ue/Variant_Combat",
			"sagad_ue/Variant_Combat/AI",
			"sagad_ue/Variant_Combat/Animation",
			"sagad_ue/Variant_Combat/Gameplay",
			"sagad_ue/Variant_Combat/Interfaces",
			"sagad_ue/Variant_Combat/UI",
			"sagad_ue/Variant_SideScrolling",
			"sagad_ue/Variant_SideScrolling/AI",
			"sagad_ue/Variant_SideScrolling/Gameplay",
			"sagad_ue/Variant_SideScrolling/Interfaces",
			"sagad_ue/Variant_SideScrolling/UI"
		});

		// Uncomment if you are using Slate UI
		// PrivateDependencyModuleNames.AddRange(new string[] { "Slate", "SlateCore" });

		// Uncomment if you are using online features
		// PrivateDependencyModuleNames.Add("OnlineSubsystem");

		// To include OnlineSubsystemSteam, add it to the plugins section in your uproject file with the Enabled attribute set to true
	}
}
