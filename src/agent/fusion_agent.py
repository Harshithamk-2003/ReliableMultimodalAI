class FusionAgent:
    """Combines all modality outputs using Dempster-Shafer evidence theory."""
    name = "Fusion Agent"

    def __init__(self, fuse_modalities_fn):
        self.fuse_modalities_fn = fuse_modalities_fn

    def fuse(self, modality_results):
        result = self.fuse_modalities_fn(modality_results)
        result['agent'] = self.name
        return result