class DataQualityAgent:
    """Checks input data before any model sees it, flags quality issues."""
    name = "Data Quality Agent"

    def check(self, modality_name, is_missing=False, is_default=False):
        flags = []
        quality = 1.0

        if is_missing:
            flags.append(f"{modality_name}: input not provided")
            quality = 0.0
        elif is_default:
            flags.append(f"{modality_name}: using default/placeholder values — reduced trust")
            quality = 0.7

        return quality, flags