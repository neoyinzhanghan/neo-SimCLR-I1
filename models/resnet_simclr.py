import torch.nn as nn
import torchvision.models as models
import torch

from exceptions.exceptions import InvalidBackboneError


class ResNetSimCLR(nn.Module):
    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {
            "resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim),
        }

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc
        )

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50"
            )
        else:
            return model

    # def get_feature(self, x):
    #     # Iterate through all layers of the backbone except the last one
    #     for layer in list(self.backbone.children())[:-1]:
    #         x = layer(x)

    #     # x now contains the features from the layer just before the last layer
    #     return x

    def forward(self, x):
        return self.backbone(x)


####################################################################################################
# BELOW ARE NEO'S NEW ADDITIONS
####################################################################################################


class SimCLRFeatureExtractor(nn.Module):
    """A Wrapper around SimCLR to extract features

    === Class Attributes ===
    - self.ckpt_path: Path to checkpoint file
    - extraction_model: a SimCLR model for which we defined a forward feature extractor. ### TODO actually I think the get_feature can be implemented here...
    """

    def __init__(self, ckpt_path: str, arch="resnet50"):
        super().__init__()

        # Load the checkpoint
        checkpoint = torch.load(ckpt_path)
        state_dict = checkpoint["state_dict"]
        # Load the model weights from the checkpoint

        for k in list(state_dict.keys()):
            if k.startswith("backbone."):
                if k.startswith("backbone") and not k.startswith("backbone.fc"):
                    # remove prefix
                    state_dict[k[len("backbone.") :]] = state_dict[k]
            del state_dict[k]

        if arch == "resnet18":
            model = models.resnet18(pretrained=False, num_classes=10)
        elif arch == "resnet50":
            model = models.resnet50(pretrained=False, num_classes=10)

        log = model.load_state_dict(state_dict, strict=False)
        assert log.missing_keys == ["fc.weight", "fc.bias"]

        model.eval()

        self.extraction_model = model


    def forward(self, x: torch.Tensor):
        # Define a hook function that captures the intermediate layer outputs
        features = {}

        def hook(module, input, output):
            features['output'] = output.detach()

        # Register the hook to the global average pooling layer or the appropriate layer before the projection head
        # Note: Adjust the attribute access (.avgpool) based on your actual model architecture
        hook_handle = self.extraction_model.avgpool.register_forward_hook(hook)

        # Perform a forward pass with the image
        # Ensure the image tensor is correctly preprocessed and moved to the same device as the model
        self.extraction_model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation for inference
            _ = self.extraction_model(x)

        # Remove the hook after getting the features to prevent memory leak
        hook_handle.remove()

        # Return the captured features
        return features['output']


    # def forward(self, x: torch.Tensor):
    #     """Forward pass"""

    #     print(list(self.extraction_model.children()))  # TODO remove, for debugging

    #     # Iterate through all layers of the backbone except the last one
    #     for layer in list(self.extraction_model.children())[:-1]:
    #         x = layer(x)

    #     # x now contains the features from the layer just before the last layer
    #     return x


def load_model(ckpt_path: str, arch="resnet50"):
    """Load an SimCLR feature extractor model from checkpoint"""

    return SimCLRFeatureExtractor(ckpt_path, arch=arch)
