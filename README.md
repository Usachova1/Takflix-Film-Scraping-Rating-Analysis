# Takflix Film Scraping & Rating Analysis

A web scraping and data analysis project that collects film metadata from the [Takflix](https://www.takflix.com) website and explores what drives film ratings — from simple text statistics (vowel/consonant counts, syllables) to a linear regression model and a content-based film recommendation system built with TF-IDF and cosine similarity.

## What This Project Does

1. **Scrapes film data** directly from Takflix using `requests` + `BeautifulSoup`: title, genre, description, rating, and runtime for every film listed across multiple search-result pages.
2. **Engineers text features** from film descriptions: vowel counts, consonant counts, syllable counts, and counts of the most frequent syllables across the corpus.
3. **Explores relationships** between these text features, runtime, and film ratings using scatter plots, histograms, t-tests, and two-way ANOVA.
4. **Builds predictive models**: a baseline linear regression (rating ~ syllable features) evaluated with Mean Squared Error and R², plus a helper function to test different numbers of top syllables as features.
5. **Builds a recommender system**: a TF-IDF vectorization of film descriptions with cosine similarity to recommend the 10 most similar films to a given title.

## Methods Used

| Stage | Technique / Library |
|---|---|
| Web scraping | `requests`, `BeautifulSoup` (HTML parsing of film listing and detail pages) |
| Data storage | `pandas` DataFrame (`films_dataframe`: Name, Genre, Description, Timing, Rating) |
| Text feature engineering | Custom vowel/consonant counters, custom Ukrainian syllable-splitting algorithm, `collections.Counter` for most-frequent syllables |
| Exploratory analysis | Seaborn scatter plots (rating vs. vowels/consonants/timing), histograms of film runtime, runtime-based dataset splits (< 40 min vs. ≥ 40 min) |
| Statistical testing | Independent-samples **t-test** (`scipy.stats.ttest_ind`) and two-way **ANOVA** (`statsmodels`, `C(most_common) + C(All_syllables) + interaction`) to test whether syllable-based features are significantly associated with rating |
| Predictive modeling | `scikit-learn` **Linear Regression** with train/test split, evaluated via **Mean Squared Error** and **R²**; a reusable `dif_syllables(n, ...)` function to compare models built on the top-*n* most common syllables |
| Recommendation system | `TfidfVectorizer` + `linear_kernel` (cosine similarity) over film descriptions to recommend similar films |

## Key Findings

- **170 films** were successfully scraped and compiled into the final dataset (out of ~200+ listing entries encountered across pages, after de-duplication and filtering).
- **Text features are statistically linked to rating.** T-tests comparing ratings against syllable-based counts returned p-values effectively at zero (e.g. p ≈ 3.5 × 10⁻⁵⁵ and p ≈ 3.6 × 10⁻⁷²), and the two-way ANOVA on `most_common` and `All_syllables` syllable counts also showed statistically significant main effects and an interaction effect (all p < 0.05) — though these tests compare raw scales (ratings ~1–5 vs. syllable counts in the tens/hundreds), so the extremely small p-values partly reflect this scale mismatch rather than a strong practical relationship.
- **The linear regression model achieved a low Mean Squared Error (≈ 0.13)** when predicting rating from syllable-based features, but since ratings cluster tightly around 4.0–4.5, this reflects the narrow rating range at least as much as genuine predictive power.
- **The TF-IDF + cosine-similarity recommender** successfully returns a ranked list of the 10 most textually similar films for a given title (demonstrated with *"Зошит війни"*), confirming the description-based similarity approach works as a basic content-based recommendation engine.
- Runtime-split scatter plots (films under vs. over 40 minutes) show that the relationship between description length/style and rating differs somewhat between short and standard-length films, motivating the runtime-based segmentation used throughout the notebook.

## Repository Structure

```
.
├── scraping_and_analyze_data_.ipynb   # main notebook: scraping, EDA, stats, ML, recommender
├── requirements.txt                    # Python dependencies
└── README.md
```

## Installation

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

1. The notebook was originally written for **Google Colab** and imports `google.colab.drive` / `google.colab.files`, though the scraping and analysis steps themselves don't require Colab-specific features. For local execution, simply remove or comment out the Colab-only imports.
2. Launch Jupyter and run the cells sequentially:

   ```bash
   jupyter notebook scraping_and_analyze_data_.ipynb
   ```

3. The scraping step (`scrape_all_film_poster`, `scrape_movie_features`) makes live HTTP requests to takflix.com and can take a few minutes depending on the number of pages (`n_max_page`) requested and network speed.

## Notes & Limitations

- Scraping depends on the current HTML structure of takflix.com; if the site's markup changes, the CSS selectors in `get_rating_from_soup`, `get_genre_from_soup`, and related functions will need to be updated.
- The `TfidfVectorizer` and `linear_kernel` imports (`sklearn.feature_extraction.text.TfidfVectorizer`, `sklearn.metrics.pairwise.linear_kernel`) are referenced in the notebook but not explicitly shown in the initial imports cell — make sure both are imported before running the recommender section.
- The syllable-splitting logic is a simplified heuristic tailored to Ukrainian vowels/consonants and won't generalize to other languages without adaptation.
- Statistical significance in this notebook should be interpreted with care: comparing raw rating values against syllable counts via a t-test conflates "different scales" with "meaningful relationship" — a correlation or regression-based effect size is more informative than the p-value alone.
- Please review Takflix's terms of service / robots.txt before running large-scale scraping, and consider adding request delays to avoid overloading the site.
