peps-detector/
│
├── peps_detector/
│   ├── __init__.py
│   ├── detector.py              # Core detection logic
│   ├── features.py              # HOG + LBP feature extraction
│   ├── utils.py                 # Helper functions
│
├── scripts/
│   ├── run_detection.py         # CLI to run detection on an image
│   ├── visualize_results.py     # Optional script to visualize detections
│
├── models/
│   └── svm_tuned.sav            # Your trained SVM model (ignored in .gitignore if private)
│
├── requirements.txt
├── README.md
├── LICENSE
└── setup.py
