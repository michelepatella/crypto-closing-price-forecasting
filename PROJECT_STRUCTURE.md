```text
.
├── data/                              # Raw cryptocurrency time-series datasets
├── models/                            # Trained model artifacts
├── reports/                           # Pipeline + experiment reports
│
├── src/    
│   ├── init.py                                                      
│   ├── config.py                      # Project configuration
│   ├── const.py                       # Global constants
│   │
│   ├── data/         
│   │   ├── init.py                          
│   │   ├── explore.py                 # Exploratory data analysis
│   │   ├── prepare.py                 # Data preparation
│   │   ├── verify_quality.py          # Data quality assurance
│   │   │
│   │   └── utils/                         
│   │       ├── adjacency_matrix.py    # Spatio-temporal graph construction
│   │       └── features.py            # Feature engineering
│   │
│   ├── modeling/        
│   │   ├── init.py                      
│   │   ├── train.py                   # T-MTGNN model training
│   │   └── evaluate.py                # Inference + metrics computation
│   │
│   └── main.py                        # End-to-end pipeline orchestration
│
├── .gitignore                         # Files/folders ignored by Git
├── .pre-commit-config.yaml            # Pre-commit hooks configuration
├── pyproject.toml                     # Project configuration, dependencies, and build settings
├── README.md                          # Project documentation
├── PROJECT_STRUCTURE.md               # Project structure overview (this file)
└── LICENSE                            # License defining project usage rights
```