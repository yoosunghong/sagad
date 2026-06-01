"""Unit checks for the variant diffusion sampler (ARCHITECT sec. 2.5)."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diffusion import (  # noqa: E402
    DiffusionMLP,
    GaussianDiffusion,
    VariantParamSpec,
    ParamField,
    building_spec,
    train_diffusion,
)


def test_param_roundtrip_and_integer_rounding():
    spec = building_spec(min_floors=2, max_floors=10)
    raw = torch.tensor([[5.0, 0.15, 0.1, 0.08],
                        [9.0, 0.30, 0.2, 0.02]])
    z = spec.to_model(raw)
    assert z.min() >= -1.0 - 1e-6 and z.max() <= 1.0 + 1e-6, "model space must be in [-1,1]"
    back = spec.from_model(z)
    assert torch.allclose(back, raw, atol=1e-5), f"round-trip mismatch:\n{back}\n{raw}"

    # out-of-range model values clamp into the valid box; floor_count stays integral
    noisy = spec.from_model(torch.tensor([[1.7, -1.9, 0.33, 0.0]]))
    assert noisy[0, 0] == 10.0, "floor_count must clamp to max and be integral"
    assert noisy[0, 1] == 0.0, "bend_gain must clamp to lo"
    assert noisy[0, 0].item().is_integer()
    print("PASS param round-trip + integer rounding")


def test_diffusion_recovers_bimodal_distribution():
    torch.manual_seed(0)
    spec = VariantParamSpec([ParamField("a", 0.0, 1.0), ParamField("b", 0.0, 1.0)])

    # two well-separated modes in model space (~ raw clusters near 0.2 and 0.8)
    n = 2000
    modeA = torch.tensor([-0.6, -0.6]) + 0.05 * torch.randn(n // 2, 2)
    modeB = torch.tensor([0.6, 0.6]) + 0.05 * torch.randn(n // 2, 2)
    data = torch.cat([modeA, modeB], dim=0)

    diff = GaussianDiffusion(DiffusionMLP(dim=2, hidden=128), dim=2, timesteps=100)
    train_diffusion(diff, data, epochs=1500, lr=1e-3, batch_size=128, log_every=750)

    diff.eval()
    samples = diff.sample(1024)
    assert torch.isfinite(samples).all()

    # assign each sample to the nearest target mode; both must be populated
    cA = torch.tensor([-0.6, -0.6])
    cB = torch.tensor([0.6, 0.6])
    dA = (samples - cA).norm(dim=1)
    dB = (samples - cB).norm(dim=1)
    near_A = (dA < dB)
    fracA = near_A.float().mean().item()
    assert 0.30 < fracA < 0.70, f"modes imbalanced (fracA={fracA:.2f}) -- mode collapse"

    mean_A = samples[near_A].mean(0)
    mean_B = samples[~near_A].mean(0)
    errA = (mean_A - cA).norm().item()
    errB = (mean_B - cB).norm().item()
    assert errA < 0.15 and errB < 0.15, f"mode means off (errA={errA:.3f}, errB={errB:.3f})"

    # decoded raw samples must respect the spec bounds
    raw = spec.from_model(samples)
    assert raw.min() >= 0.0 and raw.max() <= 1.0
    print(f"PASS bimodal recovery | fracA={fracA:.2f} errA={errA:.3f} errB={errB:.3f}")


if __name__ == "__main__":
    test_param_roundtrip_and_integer_rounding()
    test_diffusion_recovers_bimodal_distribution()
    print("ALL PASS")
