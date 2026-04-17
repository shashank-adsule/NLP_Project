# Part 1 — Working out a Toy IR System (Ambiguity Focus)

## Dataset

| Doc | Text |
|-----|------|
| d1 | The star in our solar system provides heat and light. |
| d2 | That Hollywood star walked the red carpet for the movie premiere. |
| d3 | Astronomers observe distant stars and galaxies using telescopes. |

---

## 1. Preprocessing and Inverted Index

### 1.1 Tokenization

Tokenization splits each document into individual word tokens after lowercasing.
Stop words to remove: `{the, in, our, and, that, for}`

**d1 — raw tokens:**
```
[the, star, in, our, solar, system, provides, heat, and, light]
```
After removing stop words (`the`, `in`, `our`, `and`):
```
[star, solar, system, provides, heat, light]
```

**d2 — raw tokens:**
```
[that, hollywood, star, walked, the, red, carpet, for, the, movie, premiere]
```
After removing stop words (`that`, `the`, `the`, `for`):
```
[hollywood, star, walked, red, carpet, movie, premiere]
```

**d3 — raw tokens:**
```
[astronomers, observe, distant, stars, and, galaxies, using, telescopes]
```
After removing stop words (`and`):
```
[astronomers, observe, distant, stars, galaxies, using, telescopes]
```

> **Note:** No stemming is applied here. This means `star` (d1, d2) and `stars` (d3) are treated as **different terms**. This is an important limitation that affects retrieval — discussed in Part 4.

---

### 1.2 Inverted Index

The inverted index maps each unique term to the set of documents it appears in.

| Term | Postings List |
|------|--------------|
| astronomers | {d3} |
| carpet | {d2} |
| distant | {d3} |
| galaxies | {d3} |
| heat | {d1} |
| hollywood | {d2} |
| light | {d1} |
| movie | {d2} |
| observe | {d3} |
| premiere | {d2} |
| provides | {d1} |
| red | {d2} |
| solar | {d1} |
| **star** | **{d1, d2}** |
| stars | {d3} |
| system | {d1} |
| telescopes | {d3} |
| using | {d3} |
| walked | {d2} |

> **Key observation:** `star` appears in both d1 and d2 — it is the only term shared across more than one document. All other terms are unique to a single document. This gives `star` a lower IDF weight (less discriminative), which becomes significant in the TF-IDF calculations below.

---

## 2. TF-IDF Term-Document Matrix

### 2.1 Term Frequency (TF)

TF counts how many times each term appears in each document. Since every term appears at most once per document here, all non-zero TF values are 1.

| Term | d1 | d2 | d3 |
|------|:--:|:--:|:--:|
| astronomers | 0 | 0 | 1 |
| carpet | 0 | 1 | 0 |
| distant | 0 | 0 | 1 |
| galaxies | 0 | 0 | 1 |
| heat | 1 | 0 | 0 |
| hollywood | 0 | 1 | 0 |
| light | 1 | 0 | 0 |
| movie | 0 | 1 | 0 |
| observe | 0 | 0 | 1 |
| premiere | 0 | 1 | 0 |
| provides | 1 | 0 | 0 |
| red | 0 | 1 | 0 |
| solar | 1 | 0 | 0 |
| **star** | **1** | **1** | **0** |
| stars | 0 | 0 | 1 |
| system | 1 | 0 | 0 |
| telescopes | 0 | 0 | 1 |
| using | 0 | 0 | 1 |
| walked | 0 | 1 | 0 |
| **Total terms** | **6** | **7** | **7** |

---

### 2.2 Inverse Document Frequency (IDF)

$$\text{IDF}(t) = \log\left(\frac{N}{\text{df}(t)}\right)$$

where:
- **N** = total number of documents = **3**
- **df(t)** = number of documents containing term t
- **log** = natural logarithm (base e)

| Term | df | Calculation | IDF |
|------|:--:|-------------|:---:|
| astronomers | 1 | log(3/1) | **1.0986** |
| carpet | 1 | log(3/1) | **1.0986** |
| distant | 1 | log(3/1) | **1.0986** |
| galaxies | 1 | log(3/1) | **1.0986** |
| heat | 1 | log(3/1) | **1.0986** |
| hollywood | 1 | log(3/1) | **1.0986** |
| light | 1 | log(3/1) | **1.0986** |
| movie | 1 | log(3/1) | **1.0986** |
| observe | 1 | log(3/1) | **1.0986** |
| premiere | 1 | log(3/1) | **1.0986** |
| provides | 1 | log(3/1) | **1.0986** |
| red | 1 | log(3/1) | **1.0986** |
| solar | 1 | log(3/1) | **1.0986** |
| **star** | **2** | **log(3/2)** | **0.4055** |
| stars | 1 | log(3/1) | **1.0986** |
| system | 1 | log(3/1) | **1.0986** |
| telescopes | 1 | log(3/1) | **1.0986** |
| using | 1 | log(3/1) | **1.0986** |
| walked | 1 | log(3/1) | **1.0986** |

> **Key observation:** `star` is the **only term with IDF < 1.0986** because it appears in 2 out of 3 documents (df = 2). Every other term is unique to a single document (df = 1) so they all get the maximum IDF of log(3) ≈ 1.0986. A lower IDF means the term is less useful for distinguishing between documents.

---

### 2.3 TF-IDF Weights

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

Since all TF values are 0 or 1 here, TF-IDF simply equals IDF for present terms and 0 for absent terms.

| Term | d1 | d2 | d3 |
|------|:------:|:------:|:------:|
| astronomers | 0.0000 | 0.0000 | **1.0986** |
| carpet | 0.0000 | **1.0986** | 0.0000 |
| distant | 0.0000 | 0.0000 | **1.0986** |
| galaxies | 0.0000 | 0.0000 | **1.0986** |
| heat | **1.0986** | 0.0000 | 0.0000 |
| hollywood | 0.0000 | **1.0986** | 0.0000 |
| light | **1.0986** | 0.0000 | 0.0000 |
| movie | 0.0000 | **1.0986** | 0.0000 |
| observe | 0.0000 | 0.0000 | **1.0986** |
| premiere | 0.0000 | **1.0986** | 0.0000 |
| provides | **1.0986** | 0.0000 | 0.0000 |
| red | 0.0000 | **1.0986** | 0.0000 |
| solar | **1.0986** | 0.0000 | 0.0000 |
| **star** | **0.4055** | **0.4055** | 0.0000 |
| stars | 0.0000 | 0.0000 | **1.0986** |
| system | **1.0986** | 0.0000 | 0.0000 |
| telescopes | 0.0000 | 0.0000 | **1.0986** |
| using | 0.0000 | 0.0000 | **1.0986** |
| walked | 0.0000 | **1.0986** | 0.0000 |

> **Observation:** `star` receives the **lowest TF-IDF weight (0.4055)** in both d1 and d2 because it is a shared term. All other terms that appear in only one document receive the full weight of 1.0986. This reflects TF-IDF's design intent — terms that appear in fewer documents are considered more informative.

---

### 2.4 Per-Document TF × IDF × TF-IDF Breakdown

The tables below show, for each document individually, how TF and IDF combine into the final TF-IDF weight. Only terms present in the document (TF ≥ 1) contribute a non-zero TF-IDF score.

**d1 — "The star in our solar system provides heat and light"**

| Term | TF | IDF | TF-IDF |
|------|:--:|:---:|:------:|
| heat | 1 | 1.0986 | **1.0986** |
| light | 1 | 1.0986 | **1.0986** |
| provides | 1 | 1.0986 | **1.0986** |
| solar | 1 | 1.0986 | **1.0986** |
| system | 1 | 1.0986 | **1.0986** |
| **star** | **1** | **0.4055** | **0.4055** |
| astronomers | 0 | 1.0986 | 0.0000 |
| carpet | 0 | 1.0986 | 0.0000 |
| distant | 0 | 1.0986 | 0.0000 |
| galaxies | 0 | 1.0986 | 0.0000 |
| hollywood | 0 | 1.0986 | 0.0000 |
| movie | 0 | 1.0986 | 0.0000 |
| observe | 0 | 1.0986 | 0.0000 |
| premiere | 0 | 1.0986 | 0.0000 |
| red | 0 | 1.0986 | 0.0000 |
| stars | 0 | 1.0986 | 0.0000 |
| telescopes | 0 | 1.0986 | 0.0000 |
| using | 0 | 1.0986 | 0.0000 |
| walked | 0 | 1.0986 | 0.0000 |

> d1 has 6 active terms. Five of them (`heat`, `light`, `provides`, `solar`, `system`) are unique to d1 and carry the full weight of 1.0986. `star` is shared with d2 and thus carries a reduced weight of 0.4055.

---

**d2 — "That Hollywood star walked the red carpet for the movie premiere"**

| Term | TF | IDF | TF-IDF |
|------|:--:|:---:|:------:|
| carpet | 1 | 1.0986 | **1.0986** |
| hollywood | 1 | 1.0986 | **1.0986** |
| movie | 1 | 1.0986 | **1.0986** |
| premiere | 1 | 1.0986 | **1.0986** |
| red | 1 | 1.0986 | **1.0986** |
| walked | 1 | 1.0986 | **1.0986** |
| **star** | **1** | **0.4055** | **0.4055** |
| astronomers | 0 | 1.0986 | 0.0000 |
| distant | 0 | 1.0986 | 0.0000 |
| galaxies | 0 | 1.0986 | 0.0000 |
| heat | 0 | 1.0986 | 0.0000 |
| light | 0 | 1.0986 | 0.0000 |
| observe | 0 | 1.0986 | 0.0000 |
| provides | 0 | 1.0986 | 0.0000 |
| solar | 0 | 1.0986 | 0.0000 |
| stars | 0 | 1.0986 | 0.0000 |
| system | 0 | 1.0986 | 0.0000 |
| telescopes | 0 | 1.0986 | 0.0000 |
| using | 0 | 1.0986 | 0.0000 |

> d2 has 7 active terms. Six of them (`carpet`, `hollywood`, `movie`, `premiere`, `red`, `walked`) are unique to d2 and carry the full weight of 1.0986. Again `star` is penalised to 0.4055 for being shared.

---

**d3 — "Astronomers observe distant stars and galaxies using telescopes"**

| Term | TF | IDF | TF-IDF |
|------|:--:|:---:|:------:|
| astronomers | 1 | 1.0986 | **1.0986** |
| distant | 1 | 1.0986 | **1.0986** |
| galaxies | 1 | 1.0986 | **1.0986** |
| observe | 1 | 1.0986 | **1.0986** |
| stars | 1 | 1.0986 | **1.0986** |
| telescopes | 1 | 1.0986 | **1.0986** |
| using | 1 | 1.0986 | **1.0986** |
| carpet | 0 | 1.0986 | 0.0000 |
| heat | 0 | 1.0986 | 0.0000 |
| hollywood | 0 | 1.0986 | 0.0000 |
| light | 0 | 1.0986 | 0.0000 |
| movie | 0 | 1.0986 | 0.0000 |
| premiere | 0 | 1.0986 | 0.0000 |
| provides | 0 | 1.0986 | 0.0000 |
| red | 0 | 1.0986 | 0.0000 |
| solar | 0 | 1.0986 | 0.0000 |
| star | 0 | 0.4055 | 0.0000 |
| system | 0 | 1.0986 | 0.0000 |
| walked | 0 | 1.0986 | 0.0000 |

> d3 has 7 active terms — all of them unique to d3, so every active term carries the maximum weight of 1.0986. Notably, `star` has TF = 0 here because d3 uses `stars` (plural). This is the no-stemming limitation in action.

---

## 3. Boolean Retrieval

**Query:** `"star light"`

Boolean retrieval looks up each query term in the inverted index directly.

**Step 1 — Look up each term:**

| Query Term | Postings List |
|------------|--------------|
| star | {d1, d2} |
| light | {d1} |

**Step 2 — Apply Boolean AND** (document must contain *both* terms):

```
{d1, d2}  AND  {d1}  =  {d1}
```

> **Documents matching the query: d1 only.**
> d2 contains `star` but not `light`, so it is excluded under AND.
> d3 contains neither `star` nor `light` (it has `stars`, which is a different token without stemming).

> **Note on OR retrieval:** If using Boolean OR (any term), the result would be {d1, d2}. The choice of AND vs OR significantly impacts recall vs precision.

---

## 4. Cosine Similarity and Ranking

**Query:** `"star light"`

### Step 1 — Represent the query as a TF-IDF vector

Query tokens after stopword removal: `[star, light]`
Each term gets TF = 1 in the query.

| Query Term | TF | IDF | TF-IDF |
|------------|:--:|:---:|:------:|
| star | 1 | 0.4055 | **0.4055** |
| light | 1 | 1.0986 | **1.0986** |

Query vector **q** = `{star: 0.4055, light: 1.0986}`

---

### Step 2 — Cosine Similarity Formula

$$\text{cosine}(q, d) = \frac{q \cdot d}{|q| \times |d|}$$

**Magnitudes:**

$$|q| = \sqrt{0.4055^2 + 1.0986^2} = \sqrt{0.1644 + 1.2069} = \sqrt{1.3713} = 1.1710$$

| Doc | \|d\| |
|-----|:-----:|
| d1 | √(1.0986² × 5 + 0.4055²) = √(6.0246 + 0.1644) = √6.189 = **2.4898** |
| d2 | √(1.0986² × 6 + 0.4055²) = √(7.2295 + 0.1644) = √7.394 = **2.7214** |
| d3 | √(1.0986² × 7) = √(8.4344) = **2.9067** |

---

### Step 3 — Dot products

**q · d1:**
```
star  : 0.4055 × 0.4055 = 0.1644
light : 1.0986 × 1.0986 = 1.2069
                  total = 1.3714
```

**q · d2:**
```
star  : 0.4055 × 0.4055 = 0.1644
light : 1.0986 × 0.0000 = 0.0000
                  total = 0.1644
```

**q · d3:**
```
star  : 0.4055 × 0.0000 = 0.0000
light : 1.0986 × 0.0000 = 0.0000
                  total = 0.0000
```

---

### Step 4 — Final Cosine Scores

| Doc | Dot Product | \|q\| | \|d\| | Cosine Score |
|-----|:-----------:|:-----:|:-----:|:------------:|
| d1 | 1.3714 | 1.1710 | 2.4898 | **0.4703** |
| d2 | 0.1644 | 1.1710 | 2.7214 | **0.0516** |
| d3 | 0.0000 | 1.1710 | 2.9067 | **0.0000** |

---

### Step 5 — Ranking

| Rank | Doc | Score | Document Text |
|:----:|-----|:-----:|---------------|
| 1 | **d1** | 0.4703 | The star in our solar system provides heat and light. |
| 2 | d2 | 0.0516 | That Hollywood star walked the red carpet for the movie premiere. |
| 3 | d3 | 0.0000 | Astronomers observe distant stars and galaxies using telescopes. |

> **Final ranking: d1 > d2 > d3**

> **Is the ranking desirable?**
> **Yes, largely.** d1 is correctly placed at rank 1 — it contains both `star` (the Sun) and `light`, making it a strong match for the astronomical query "star light". d2 ranks second only because it shares the token `star` (a celebrity), contributing a small cosine score of 0.0516. d3 scores zero despite being about astronomy, because it uses the plural form `stars` instead of `star` — without stemming these are different tokens. This reveals a real limitation: the system misses d3 entirely, which is arguably more relevant to astronomy than d2.

---

## 5. Analysis of Word Sense Ambiguity

**Query:** `"movie star"`

### Step 1 — Query TF-IDF vector

| Query Term | TF | IDF | TF-IDF |
|------------|:--:|:---:|:------:|
| movie | 1 | 1.0986 | **1.0986** |
| star | 1 | 0.4055 | **0.4055** |

Query vector **q** = `{movie: 1.0986, star: 0.4055}`

---

### Step 2 — Cosine Scores

| Doc | Dot Product | Cosine Score |
|-----|:-----------:|:------------:|
| d1 | star: 0.4055 × 0.4055 = 0.1644 → total = **0.1644** | **0.0564** |
| d2 | movie: 1.0986 × 1.0986 = 1.2069 + star: 0.4055 × 0.4055 = 0.1644 → total = **1.3713** | **0.4303** |
| d3 | 0.0000 | **0.0000** |

### Step 3 — Ranking

| Rank | Doc | Score | Document Text |
|:----:|-----|:-----:|---------------|
| 1 | **d2** | 0.4303 | That Hollywood star walked the red carpet for the movie premiere. |
| 2 | d1 | 0.0564 | The star in our solar system provides heat and light. |
| 3 | d3 | 0.0000 | Astronomers observe distant stars and galaxies using telescopes. |

---

### Step 4 — Word Sense Ambiguity Analysis

> **Which document should ideally be retrieved?**
> **d2 only.** The query "movie star" clearly refers to a celebrity in the entertainment industry. d2 is the only document about Hollywood, movies, and celebrity culture — it is the correct and complete answer.

> **Does the system retrieve the correct document?**
> **Yes, partially.** d2 is correctly ranked first with score 0.4303. However, d1 is also retrieved at rank 2 with score 0.0564, which is incorrect — d1 is about the Sun and has no connection to movies or celebrities.

> **How does word sense ambiguity affect retrieval?**
> The word `star` carries two completely different meanings:
> - **Sense 1 (astronomical):** A luminous celestial body — relevant to d1 and d3
> - **Sense 2 (entertainment):** A famous celebrity or actor — relevant to d2
>
> The VSM (Vector Space Model) has **no concept of word meaning**. It treats `star` as a single token regardless of context. So when the query is "movie star", the model sees `star` as a shared feature between d1 and d2, and d1 receives a non-zero score (0.0564) purely because it contains the same token — even though d1 is about the Sun and is completely irrelevant to the query's intent.
>
> The fundamental problem is that **the same string maps to two different concepts**, and the bag-of-words model cannot distinguish between them. A semantically-aware model such as **LSA (Latent Semantic Analysis)** or **word embeddings (Word2Vec, BERT)** would capture that in the context of "movie", the word "star" is semantically closer to "celebrity", "actor", "Hollywood" — and would score d2 much higher while suppressing d1. This is one of the core motivations for improving beyond the basic VSM in Part 5.

---

## Summary of All Results

| Question | Answer |
|----------|--------|
| Unique terms in corpus | 19 |
| Only shared term | `star` (d1, d2) — IDF = 0.4055 |
| All other terms IDF | 1.0986 (appear in exactly 1 doc) |
| Boolean AND for "star light" | **{d1}** |
| Boolean OR for "star light" | {d1, d2} |
| Cosine ranking for "star light" | **d1 (0.4703) > d2 (0.0516) > d3 (0.0000)** |
| Cosine ranking for "movie star" | **d2 (0.4303) > d1 (0.0564) > d3 (0.0000)** |
| Does "star light" ranking make sense? | Yes — d1 is correctly first; d3 missed due to no stemming |
| Does "movie star" retrieve correctly? | Mostly — d2 is first but d1 is incorrectly included |
| Root cause of d1 appearing in "movie star" | Word sense ambiguity — VSM cannot differentiate senses of `star` |