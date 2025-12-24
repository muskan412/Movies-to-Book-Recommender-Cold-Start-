# Semantic Bridges: LLM-Based Cross-Domain Recommendation from Movies to Books

This repository contains the code and artifacts for our AML course project:

> **Goal:** given only a user’s *movie* taste, recommend *books* they are likely to enjoy – without any shared users, joint ratings, or cross-domain IDs.

We build a semantic “bridge” between Rotten Tomatoes movies and Amazon books using sentence-level LLM embeddings, then compare:

1. **In-domain baseline:** Neural Collaborative Filtering (NCF) on book-only ratings
2. **Approach A – Zero-shot semantic retrieval:** movie and book embeddings in a shared space
3. **Approach B – Learned mapping:** a small MLP trained with triplet loss to map movie embeddings into book space

The entire pipeline is implemented in a single Jupyter notebook (`AML_Final_Project_updated_with_outputs.ipynb`).

## 1. Problem & Setting

Traditional recommenders assume you already have **user–item interaction history in one domain** (e.g., users rating books on Goodreads). They break in two common situations:

* **User cold start:** a new user has no ratings in the target system
* **Cross-domain cold start:** a user has history in one domain (movies) but none in another (books)

In our project, we tackle the second case:

> **Given only the movies a user likes, recommend books – with no overlapping users, no joint ratings, and no shared IDs between movie and book catalogs.**

Collaborative filtering cannot even “see” such a user, because they have not rated any books.
Instead, we rely on **text** (plots, reviews, categories, metadata) and **pretrained LLM embeddings** to embed movies and books into a shared semantic space and then retrieve cross-domain neighbors.

## 2. Data

All datasets are public Kaggle CSVs intended for academic use.

### 2.1 Movie data – Rotten Tomatoes

* Data: [`rotten_tomatoes_movies.csv`](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset)
* Key fields (after trimming):

  * `movie_title`
  * `movie_info` (plot synopsis)
  * `critics_consensus`
  * `genres`
  * `directors`
  * `authors` (for adaptations)
  * `tomatometer_rating`, `tomatometer_count`
  * `audience_rating`, `audience_count`

### 2.2 Book metadata – Amazon Books

* Data: https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews
* Key fields:

  * `Title`
  * `description`
  * `authors`
  * `categories` (subject / genre)

### 2.3 Book reviews & ratings – Amazon Books Reviews

* Data: https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews
* Key fields:

  * `Title`
  * `review/text`
  * `review/summary`
  * `review/score` (1–5)
  * `review/helpfulness` (e.g., `"3/5"`)

### 2.4 Basic preprocessing & joins

* Keep only books with at least 5 reviews; cap at 20 reviews per book for scalability
* Merge metadata and reviews on `Title`
* Assign integer IDs:

  * `movie_id` for each movie
  * `book_id` for each distinct book title
  * `review_id` for each review row
* Trim unused technical/UI columns

## 3. Environment & Dependencies

### 3.1 Python version

* **Python 3.10** (project was developed with `pyenv` on 3.10.12)

> Note: many scientific packages in the notebook assume **NumPy 1.x**; if you install NumPy 2.x you may hit ABI errors with some libraries (e.g., UMAP, TensorFlow). If that happens, pin:
>
> ```bash
> pip install "numpy<2.0"
> ```

### 3.2 Core libraries

Install with `pip` (preferred) or `conda`. Example:

```bash
pip install \
  numpy \
  pandas \
  scikit-learn \
  sentence-transformers \
  torch \
  tqdm \
  matplotlib \
  seaborn \
  umap-learn
```

**Used packages (as seen in the notebook):**

* **Data handling**

  * `pandas` – CSV loading, joins, groupby aggregation
  * `numpy` – numerical arrays, similarity computations

* **Text & embeddings**

  * `sentence-transformers` – `SentenceTransformer("all-mpnet-base-v2")` for 768-d sentence embeddings
  * `re`, `string` – regex-based cleaning

* **Modeling**

  * **PyTorch** (`torch`, `torch.nn`, `torch.optim`, `torch.utils.data`)

    * Custom `Dataset` + `DataLoader` for synthetic movie–book pairs
    * MLP mapping network with `TripletMarginLoss`
  * **scikit-learn**

    * `train_test_split` – splitting synthetic pairs
    * `LabelEncoder` – encoding user and book IDs for NCF
    * `TSNE` – t-SNE visualization of the shared space
    * `pairwise.cosine_similarity` – similarity matrices

* **Visualization**

  * `matplotlib.pyplot` – bar charts, histograms, similarity distributions
  * `seaborn` – density plots, KDE overlays

* **Utilities**

  * `tqdm.auto` – progress bars
  * `random` – negative sampling and subsampling
  * `os`, `pathlib` – file paths
  * `json`, `pickle` – saving and loading intermediate artifacts

## 4. Pipeline Overview

The project is structured into phases inside the notebook.

### Phase 1 – Data loading, cleaning, and exploration

* Load the three CSV files into pandas
* Trim to relevant columns and drop obvious noise
* Filter books by review count (≥ 5, max 20 per title)
* Assign IDs (`movie_id`, `book_id`, `review_id`)
* Produce basic EDA plots:

  * **Top-10 book categories** (bar chart)
  * **Top-10 movie genres** (bar chart)

### Phase 2 – Text preprocessing & semantic components

* LLM-aware cleaning:

  * lowercase, strip HTML and URLs
  * remove emojis and noisy symbols
  * collapse whitespace, but keep natural language intact

* Convert ratings into short text phrases (e.g., “very highly rated”)

* Build **semantic bridge fields**:

  **Movies**

  * `E_plot` – movie_info (plot)
  * `E_consensus` – critics_consensus + audience cues
  * `E_metadata` – genres, directors, authors
  * `E_ratings` – textualized rating signal

  **Books**

  * `E_reviews` – weighted aggregate of review embeddings (helpfulness-weighted)
  * `E_description` – publisher description (when available)
  * `E_metadata` – categories
  * `E_ratings` – rating text

* Embed each component with `SentenceTransformer("all-mpnet-base-v2")`

### Phase 3 – Weighted fusion into final item embeddings

* Combine components into a single normalized vector per item:

  **Movies**

  * `E_plot`: 0.50
  * `E_consensus`: 0.30
  * `E_metadata`: 0.10
  * `E_ratings`: 0.10

  **Books**

  * `E_reviews`: 0.45
  * `E_description`: 0.25
  * `E_metadata`: 0.10
  * `E_ratings`: 0.20

* Implemented via a helper `fuse_components_weighted(...)` which:

  * collects available component vectors
  * applies weights
  * L2-normalizes the result

* Save fused embeddings to `.npz` / `.pkl` under `artifacts/` so later phases do not re-embed.

### Phase 4 – In-domain collaborative baseline: Book NCF

* Build an **NCF** model in the **book domain only**:

  * GMF + MLP style architecture (`user_id`, `book_id` as learnable embeddings)
  * Trained on real user–book ratings from `Books_rating.csv`

* Evaluation protocol:

  * For each held-out (user, positive_book):

    * Sample 99 random negative books
    * Score all 100 with NCF
    * Check whether the positive appears in the top-10 (HitRate@10)

* Findings:

  * HitRate@10 ≈ **0.24** in the 1-of-99 setup
  * HitRate@10 ≈ **0** when asked to rank the full ~78k catalog
  * NCF **cannot** use movie-only input; it is included purely as an in-domain reference.

### Phase 5 – Cross-domain semantic recommenders

#### 5.1 Approach A – Zero-shot semantic retrieval

* Use **fused embeddings only** (no extra training):

  * For a user:

    * collect embeddings of liked movies
    * compute their mean vector and L2-normalize → user vector `u`
  * For each book:

    * use fused embedding `E_book[i]`
  * Score by cosine similarity: `score[i] = u · E_book[i]`
  * Rank all books and take top-K

* System-level evaluation:

  * Build ~48k **synthetic movie–book pairs** by taking each movie and its nearest neighbor books in the zero-shot space

  * For each pair:

    * use the movie as query
    * check whether the paired book appears in top-K

  * Metrics computed:

    * `HitRate@10`, `NDCG@10`, `Recall@10`, `Precision@10`, `GenreMatch@1`

#### 5.2 Approach B – Mapping network (MLP + Triplet Loss)

* Generate synthetic alignment pairs as above

* Train a small **MLP mapper** in PyTorch:

  ```text
  Movie embedding (768-d)
      → Linear(768 → 512) + ReLU
      → Linear(512 → 768)
      → L2 normalize → mapped movie in book space
  ```

* Training:

  * `TripletMarginLoss(margin=0.2)`
  * Adam optimizer, 9 epochs
  * Triplets: (anchor = movie, positive = synthetic aligned book, negative = less similar book for the same movie)

* After training:

  * Map all movies → `E_movie_mapped`
  * Re-run the same system-level evaluation as zero-shot
  * Use mapped user vectors for user-level recommendations (plus genre-aware re-ranking and category filters).

## 5. Evaluation & Key Results

### 5.1 In-domain NCF baseline (books only)

* HitRate@10 ≈ **0.24** for 1-of-99 book recommendation
* Collapses when forced to rank full catalog
* Cannot operate in movie→book cross-domain cold-start setting

### 5.2 System-level movie→book retrieval (synthetic pairs)

On ~48k synthetic movie–book pairs and full-catalog ranking:

* **Approach A – Zero-shot**

  * HitRate@10 ≈ **0.4625**
  * NDCG@10 ≈ **0.2361**
  * GenreMatch@1 ≈ **0.35**

* **Approach B – Mapped**

  * HitRate@10 ≈ **0.3055**
  * NDCG@10 ≈ **0.1666**
  * GenreMatch@1 ≈ **0.4060**

Random-pair sanity check:

* Random movie–book cosine similarity (zero-shot space):

  * mean ≈ **0.26**, 95th percentile ≈ **0.44**
* Aligned evaluation pairs (after mapping):

  * mean ≈ **0.56**, most above **0.4**

And for random pairs in the **mapped** space:

* Zero-shot random mean ≈ **0.266**
* Mapped random mean ≈ **0.074**

Interpretation: the mapper keeps true pairs close, but pushes **unrelated** movie–book pairs near “no relation,” increasing contrast between signal and noise.

### 5.3 User-level Nolan-style case study

* Profile: 26 high-level movies (e.g., “The Dark Knight”, “Inception”, “Interstellar”)

* Coarse movie genre footprint:

  * `{science, fantasy, adventure, mystery, suspense, fiction, action, drama}`

* Using mapped + refined retrieval:

  * Top-20 books mostly crime/thriller fiction, sci-fi, graphic novels
  * Genre overlap ≈ **0.70**
  * Average book rating ≈ **4.18/5** vs global ≈ **4.27/5**

Zero-shot and mapped lists both look genre-consistent; mapping mainly adds more structured control via re-ranking and category filters.

## 6. Running the Notebook

1. **Clone / copy the repo** and place the three CSVs into `data/`.

2. **Create a virtual environment** (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Lab / Notebook:**

   ```bash
   jupyter lab
   ```

4. **Open `AML_Final_Project.ipynb`** and run cells in order.

   * You can toggle between:

     * Full pipeline (all movies & books) – slower, but reproduces main results
     * Smaller subsets (if you uncomment the sampled versions in the notebook) – faster for debugging

5. Intermediate artifacts (embeddings, synthetic pairs, mapped vectors, recommendation pickles) will be written to `artifacts/` as configured in the notebook.

## 7. Reuse & Extension

You can adapt this code to other cross-domain settings by:

* Swapping in other text-heavy domains (e.g., games → books, podcasts → books)
* Replacing `all-mpnet-base-v2` with newer LLM embedding models
* Feeding real cross-domain interaction data into the mapping network instead of synthetic pairs
* Using FAISS for large-scale similarity search over millions of items
