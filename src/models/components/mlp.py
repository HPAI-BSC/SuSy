import torch
from torch import nn
from typing import List


class MLP(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        in_features: int,
        intermediate_features: int,
        methods: List[str],
        multiclass: bool = True,
        extract_features: bool = False,
    ) -> None:
        
        super().__init__()
        self.output_size = len(methods) if multiclass else 2
        self.fc1 = nn.Linear(in_features, intermediate_features)
        self.fc2 = nn.Linear(intermediate_features, intermediate_features)
        self.fc3 = nn.Linear(intermediate_features, self.output_size)
        self.dropout = nn.Dropout(0.5)

        self.extract_features = extract_features
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        if self.extract_features:
            return x
        x = self.dropout(x)
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    _ = MLP()
