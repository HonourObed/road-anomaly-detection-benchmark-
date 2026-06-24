# Dataset Sources

This study uses a merged corpus of two publicly available road anomaly datasets.

## Source 1: Pothole Detection Dataset

**Citation:** Rajdalsaniya. (2022). Pothole Detection Dataset (Version 1). Kaggle.
https://www.kaggle.com/datasets/rajdalsaniya/pothole-detection-dataset

- 2,077 annotated images
- Single class: Pothole
- Origin: Analytics Vidhya Dataverse Hackathon

**Download:**
```bash
kaggle datasets download -d rajdalsaniya/pothole-detection-dataset
```

## Source 2: speed-unmarked-bumb Dataset

**Citation:** Pothole Detection. (2023). speed-unmarked-bumb Dataset (Version 3). Roboflow Universe.
https://universe.roboflow.com/pothole-detection-1nczj/speed-unmarked-bumb

- Original 5 classes: Potholes, Manholes, Open Manholes, Speed Bumps, Unmarked Bumps
- Classes retained in this study: Pothole, Speed Bump
- Manhole, Open Manhole, and Unmarked Bump annotations were filtered out at the Roboflow workspace level


## Merged Corpus

| Split      | Images | Proportion |
|------------|--------|------------|
| Train      | 3,111  | 70%        |
| Validation | 667    | 15%        |
| Test       | 666    | 15%        |
| **Total**  | **4,444** |         |

**Preprocessing:** Auto-orientation applied, resized to 640x640 (stretch). No augmentations applied at dataset level.

**Classes:** 2 (Pothole, Speed Bump)

**All stress testing was conducted on the validation split only (667 images). The test split was held out throughout.**
