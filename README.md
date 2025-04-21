# contextual-sentiment-yelp
### Context-Aware Sentiment Analysis of Yelp Reviews using Fine-Tuned DistilBERT

**Tech Stack:** **Python**, **Transformers (Hugging Face)**, **DistilBERT**, **Pandas**, **Scikit-learn**, **Matplotlib/Seaborn**, **TF-IDF**, **WordCloud**

This project investigates what makes a Yelp review "positive" or "negative" beyond its star rating—by analyzing review text along with contextual metadata such as business maturity, user engagement, and rating behavior. A pre-trained **DistilBERT** model is fine-tuned on a stratified sample of Yelp reviews, with structured metadata incorporated into the input via custom special tokens.

The pipeline includes:
- **Exploratory data analysis** to identify structure and bias in review metadata  
- **Context-aware fine-tuning** of a DistilBERT model using stratified sampling and metadata augmentation  
- **Inference at scale**, comparing fine-tuned predictions to a baseline model  
- **Thematic analysis** using TF-IDF and word cloud generation  

---

### Dataset

This project uses the [Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset/), which contains reviews, user profiles, and business information across various categories.

The data used here is a minimally preprocessed subset of the original dataset, filtered to include only restaurant reviews from a cleaned business list. Preprocessing steps—performed outside this repository—included:
- Filtering reviews based on restaurant business types
- Light cleanup of metadata (e.g., column renaming, value standardization)

All analysis in this repository builds on top of this filtered data, starting from `00_EDA_Yelp_Reviews.ipynb`.

---

### Project Files

#### `00_EDA_Yelp_Reviews.ipynb`
- Loads and cleans Yelp review data, including basic text preprocessing and feature engineering.
- Performs exploratory data analysis on review metadata (e.g., review length, star ratings, engagement indicators).
- Informs the design of contextual tokens used in fine-tuning.

#### `01_WorkingData_Setup.ipynb`
- Merges Yelp review, user, and business datasets to construct a unified modeling dataset.
- Generates derived features (e.g., rating stability, business maturity) and performs fuzzy matching on city names.
- Outputs a processed dataset (`reviews_working.csv`) for use in sentiment classification with the fine-tuned model.

#### `02_FineTune_DistilBERT.ipynb`
- Fine-tunes a pre-trained **DistilBERT** model from Hugging Face on a stratified 10% sample of Yelp reviews.
- Augments review text with structured metadata tokens:
  - `[PRE_2020]`, `[POST_2020]` (review timing)
  - `[USEFUL]`, `[FUNNY]`, `[COOL]` (engagement indicators)
  - `[ESTABLISHED]`, `[NEW_BUSINESS]` (business maturity)
  - `[HIGH_STABILITY]`, `[LOW_STABILITY]` (user rating behavior)
  - `[STATE]` (location token)
- Implements class-weighted training using Hugging Face's `Trainer` API to address label imbalance.
- Evaluates model using stratified train/val/test split with metrics such as accuracy, F1, precision, recall, and confusion matrices.
- Saves the fine-tuned model and tokenizer for downstream inference.

#### `03_Apply_FinetunedDB_FullYelp.ipynb`
- Applies the fine-tuned DistilBERT model to the full Yelp dataset for sentiment inference.
- Compares results to baseline predictions and highlights reviews with the largest discrepancies.
- Generates TF-IDF-based word clouds and extracts distinctive textual themes from key review segments.

---

### Key Takeaways
- **Context improves performance**: Incorporating metadata (e.g., business maturity, engagement, time) into review text boosted DistilBERT’s F1 from 0.74 to 0.95 compared to the baseline.
- **Fine-tuned model detects nuance**: The model better captured subtle negativity and mixed-tone reviews that were often misclassified by the baseline.
- **Structured metadata is valuable**: Metadata tokens provided lightweight but meaningful context, demonstrating how structured data can enhance language model outputs.
- **TF-IDF analysis adds interpretability**: Word cloud generation and top-term extraction helped highlight what users emphasize in positive versus negative reviews across different segments.
