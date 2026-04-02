# DM's hierarchy setup for the AHP model

CATEGORIES = ["Economic", "Social/Health", "Geography/Environment"]

CATEGORY_CRITERIA = {
    "Economic":              ["Employment rate", "Long-term unemployment rate", "Personal earnings"],
    "Social/Health":         ["Life expectancy", "Life satisfaction", "Employees working very long hours"],
    "Geography/Environment": ["Air pollution", "Distance from Poznan (km)"],
}

CRITERION_CATEGORY = {
    crit: cat
    for cat, crit_list in CATEGORY_CRITERIA.items()
    for crit in crit_list
}

CRITERIA = [
    crit
    for cat in CATEGORIES
    for crit in CATEGORY_CRITERIA[cat]
]

DIRECTIONS = {
    "Employment rate":                   +1,  # gain
    "Long-term unemployment rate":       -1,  # cost
    "Personal earnings":                 +1,  # gain
    "Life expectancy":                   +1,  # gain
    "Life satisfaction":                 +1,  # gain
    "Employees working very long hours": -1,  # cost
    "Air pollution":                     -1,  # cost
    "Distance from Poznan (km)":         -1,  # cost
}