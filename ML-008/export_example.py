"""Export a tiny policy to exercise the TorchScript interface before training."""

import argparse
from pathlib import Path

import torch


class ExamplePolicy(torch.nn.Module):
    @torch.jit.export
    def reset(self, tram_id: int) -> None:
        # A feed-forward policy has no memory to clear.
        return

    @torch.jit.export
    def act(self, tram_id: int, obs: torch.Tensor) -> int:
        if obs[3] > 0:
            return 0
        if obs[0] < obs[1] - 0.3:
            return 2
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("example_policy.pt"))
    args = parser.parse_args()
    torch.jit.script(ExamplePolicy().eval()).save(str(args.output))
