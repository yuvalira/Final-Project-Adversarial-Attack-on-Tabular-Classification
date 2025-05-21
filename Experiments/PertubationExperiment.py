import numpy as np

def gaussian_perturb(category_scores: dict, current_category: str, sigma=None, sigma_scale=1.0):
    """
    Perturb a category based on correlation scores using Gaussian sampling.

    Args:
        category_scores: dict of {category_name: correlation_score}
        current_category: the category to perturb
        sigma: standard deviation for Gaussian; if None, will compute from median distance
        sigma_scale: scale factor for auto sigma (default 1.0)

    Returns:
        new_category: the perturbed category
        new_score: the correlation score of the perturbed category
    """
    if current_category not in category_scores:
        raise ValueError(f"'{current_category}' not found in category scores.")

    categories = list(category_scores.keys())
    positions = np.array([category_scores[cat] for cat in categories])

    center = category_scores[current_category]

    # Auto sigma based on median pairwise distance if not provided
    if sigma is None:
        diffs = np.abs(positions[:, None] - positions[None, :])
        median_dist = np.median(diffs[np.triu_indices_from(diffs, k=1)])  # exclude diagonal
        sigma = sigma_scale * median_dist
        if sigma == 0:
            sigma = 1e-6  # avoid division by zero

    distances = positions - center
    weights = np.exp(-0.5 * (distances / sigma) ** 2)
    weights /= weights.sum()

    sampled_index = np.random.choice(len(categories), p=weights)
    new_category = categories[sampled_index]
    return new_category, category_scores[new_category]



# Your feature dictionaries
relationship = {
    "Husband": 0.4521,
    "Wife": 0.1375,
    "Other-relative": -0.1199,
    "Unmarried": -0.1918,
    "Not-in-family": -0.2512,
    "Own-child": -0.2949
}

native_country = {
    "United-States": 0.0463,
    "France": 0.0200,
    "Germany": 0.0196,
    "Canada": 0.0172,
    "Philippines": 0.0167,
    "Italy": 0.0154,
    "Yugoslavia": 0.0142,
    "Iran": 0.0134,
    "China": 0.0110,
    "Japan": 0.0090,
    "India": 0.0077,
    "Ecuador": 0.0061,
    "Taiwan": 0.0056,
    "Ireland": 0.0040,
    "Greece": 0.0035,
    "South": 0.0029,
    "England": 0.0020,
    "Cuba": 0.0000,
    "Cambodia": -0.0040,
    "Laos": -0.0047,
    "Poland": -0.0056,
    "Scotland": -0.0061,
    "Trinadad&Tobago": -0.0061,
    "Hong": -0.0061,
    "Portugal": -0.0075,
    "Thailand": -0.0086,
    "Haiti": -0.0096,
    "Hungary": -0.0149,
    "Outlying-US(Guam-USVI-etc)": -0.0149,
    "Vietnam": -0.0169,
    "Puerto-Rico": -0.0180,
    "Jamaica": -0.0180,
    "El-Salvador": -0.0208,
    "Dominican-Republic": -0.0223,
    "Guatemala": -0.0223,
    "Nicaragua": -0.0224,
    "Columbia": -0.0331,
    "Peru": -0.0350,
    "Mexico": -0.0839
}

workclass = {
    "Self-emp-inc": 0.1288,
    "Federal-gov": 0.0808,
    "Local-gov": 0.0494,
    "Self-emp-not-inc": 0.0253,
    "State-gov": 0.0169,
    "Without-pay": -0.0149,
    "Private": -0.1465
}

# Try a few perturbations
cat1, score1 = gaussian_perturb_category(relationship, "Wife", sigma=0.07)
cat2, score2 = gaussian_perturb_category(native_country, "France", sigma=0.07)
cat3, score3 = gaussian_perturb_category(workclass, "Private", sigma=0.07)

print("")
print(f"'Wife' perturbed to '{cat1}'")
print(f"'France' perturbed to '{cat2}'")
print(f"'Private' perturbed to '{cat3}'")


print("\nAuto Adaptive Sigma:\n")
cat4, score4 = gaussian_perturb_category(relationship, "Wife", sigma=None)
cat5, score5 = gaussian_perturb_category(native_country, "France", sigma=None, sigma_scale=0.5)
cat6, score6 = gaussian_perturb_category(workclass, "Private", sigma=None, sigma_scale=1)

print(f"'Wife' perturbed to '{cat4}'")
print(f"'France' perturbed to '{cat5}'")
print(f"'Private' perturbed to '{cat6}'")

# You can also control the spread using sigma_scale
