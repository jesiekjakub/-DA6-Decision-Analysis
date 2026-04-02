# src/ahp/dm_matrices.py
import numpy as np

# Goal: Categories
# Order: [Economic, Social/Health, Geography/Environment]
# Inconsistent judgments introduced here: Economic=5xSocial, Social=2xGeo implies Economic=10xGeo, but DM said 3x.
A_goal = np.array([
    [1,    5,    3  ],
    [1/5,  1,    2  ],
    [1/3,  1/2,  1  ],
], dtype=float)

# Economic: Criteria
# Order: [Employment rate, LT unemployment rate, Personal earnings]
A_economic = np.array([
    [1,    1/2,  1/5],
    [2,    1,    1/5],
    [5,    5,    1  ],
], dtype=float)

# Social/Health: Criteria
# Order: [Life expectancy, Life satisfaction, Working long hours]
A_social = np.array([
    [1,    1/3,  1/2],
    [3,    1,    2  ],
    [2,    1/2,  1  ],
], dtype=float)

# Geography: Criteria
# Order: [Air pollution, Distance from Poznan]
A_geography = np.array([
    [1,    1/3],
    [3,    1  ],
], dtype=float)

MATRICES = {
    "Goal":                  A_goal,
    "Economic":              A_economic,
    "Social/Health":         A_social,
    "Geography/Environment": A_geography,
}