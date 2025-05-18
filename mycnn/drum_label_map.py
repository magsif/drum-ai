class DrumLabelMap:
    categories = ["kick", "snare", "hihat", "tomtom", "cymbal"]
    label_map = {category: idx for idx, category in enumerate(categories)}