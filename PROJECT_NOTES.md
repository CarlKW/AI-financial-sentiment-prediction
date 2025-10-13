# AI Financial Sentiment Prediction - Project Notes & Decisions

**Project:** SSY340 - Financial Sentiment Analysis for Stock Market Prediction  
**Date Started:** October 2025  
**Goal:** Predict stock price movements using Twitter sentiment analysis

---

## 1. DATA PREPROCESSING & CLEANING

### 1.1 Text Cleaning Strategy

#### Decision: Keep Digits and Financial Symbols
**Question:** "Is it not good to keep digits? For determining if the sentiment is good or bad?"

**Answer:** YES - Digits are crucial for financial sentiment!
- **Examples:**
  - "stock up 20%" vs "stock up 5%" - magnitude matters!
  - "price target $500" vs "price target $100" - very different signals
  - "missed earnings by 10%" - negative sentiment with context

**Implementation:**
```python
# Keep letters, digits, and financial symbols (%, $, .)
text = re.sub(r'[^a-z0-9\s%$._]', '', text)
```

**Trade-offs:**
- ✅ Preserves financial context and magnitude
- ✅ "earnings beat 15%" != "earnings beat"
- ❌ Some noise from random numbers
- ❌ Model might overfit to specific numbers

**Decision:** KEEP digits - financial domain requires them.

---

#### Decision: Preserve Stock Tickers
**Question:** "Often times stocks are abbreviated with capital letters, AMZN for example. How does removing capital letters influence the model?"

**Answer:** CRITICAL ISSUE - Tickers are key identifiers!
- **Problem:** Converting "AMZN" → "amzn" loses ticker signal
- **Solution:** Mark tickers before lowercasing

**Implementation:**
```python
# Identify stock ticker patterns (all caps, 2-5 letters)
text = re.sub(r'\b([A-Z]{2,5})\b', r'ticker_\1', text)
# Then lowercase: "AMZN" → "ticker_amzn"
text = text.lower()
```

**Benefits:**
- ✅ Preserves ticker identity: "AMZN" vs "Amazon"
- ✅ Model can learn ticker-specific patterns
- ✅ Works across case variations: "AMZN", "$amzn", "#AMZN"
- ✅ Consistent representation

**Trade-offs:**
- ❌ Slight risk of false positives (e.g., "THAT", "MAKE")
- ✅ But improves ticker recognition significantly

**Decision:** PRESERVE tickers with "ticker_" prefix.

---

#### Decision: New Column vs Overwriting
**Question:** "Why add a column [cleaned_text] instead of overwriting [Text]?"

**Answer:** Keep both for debugging and validation!

**Reasons:**
1. **Quality Control:** Compare original vs cleaned
   ```python
   df[['Text', 'cleaned_text']].head()
   # Verify cleaning didn't remove important info
   ```

2. **Debugging:** Find over-aggressive cleaning
   ```python
   # Find tweets that became too short
   df[df['cleaned_text'].str.len() < 5][['Text', 'cleaned_text']]
   ```

3. **Flexibility:** Different models might need different preprocessing
   - BERT models: might use original text
   - Traditional ML: use cleaned_text
   - Ensemble: compare both

4. **Reproducibility:** Can always regenerate if needed

**Storage cost:** Minimal (~10MB for 60k tweets)

**Decision:** KEEP both columns - data is cheap, debugging is expensive.

---

### 1.2 Feature Engineering Decisions

#### Decision: `pct_change()` for Daily Returns
**Function:** `stock_prices['daily_return'] = stock_prices.groupby('Stock Name')['Close'].pct_change()`

**What it does:**
- Calculates: `(today_price - yesterday_price) / yesterday_price`
- Example: $100 → $110 = 0.10 (10% gain)

**Why `groupby('Stock Name')` is critical:**
- WITHOUT: Would compare TSLA day 10 with AMZN day 11 (nonsense!)
- WITH: Each stock's returns calculated independently
- First day of each stock: NaN (no previous day)

**Usage:** Correlate tweet sentiment with next-day returns

---

#### Decision: `.apply()` Method
**Usage:** `df['cleaned_text'] = df['Text'].apply(clean_text)`

**How it works:**
- Applies function to EVERY row individually
- Like: `for row in df['Text']: clean_text(row)`
- But faster (vectorized) and cleaner syntax

**Why use it:**
- ✅ Clean, readable code
- ✅ Pandas-optimized
- ✅ Easy to debug (test function on one row first)
- ✅ Preserves DataFrame structure

---

## 2. EXPLORATORY DATA ANALYSIS (EDA)

### 2.1 Sentiment Class Imbalance

**Finding:** 1.75:1 ratio (3,685 positive vs 2,106 negative)

**Question:** "There is a class imbalance warning. But since it's just sentiment classification, and manual at that, it should not be a problem?"

**Answer:** CORRECT - This level is manageable!

**Analysis:**
- Ratios < 2:1 are generally okay
- NOT like fraud detection (1:1000 imbalance)
- Could genuinely reflect Twitter's positive bias
- Financial Twitter tends toward optimism (people share wins)

**Strategy:**
- ✅ Start without balancing
- ✅ Use F1-score (not accuracy) for evaluation
- ✅ Monitor per-class performance (precision/recall)
- ⚠️ If negative class recall < 0.7, then consider class_weight='balanced'

**Decision:** NO immediate action - monitor during training.

---

### 2.2 Stock Imbalance - CRITICAL ISSUE

**Finding:** Tesla (TSLA) has 29,938 tweets (47.1% of data!)
- TSLA: 29,938 tweets
- TSM: 7,528 tweets (11.9%)
- Median: 515 tweets
- **TSLA has 58x more data than median stock!**

**Why this is a BIG problem:**
- Model will learn "Tesla-speak" not general sentiment
- Tesla has unique characteristics:
  - Elon Musk mentions
  - EV/tech specific language
  - Meme stock behavior
  - Different volatility patterns
- Performance metrics will be Tesla-dominated
- Poor generalization to other stocks

**More serious than sentiment imbalance!**

---

### 2.3 Stock-Specific Sentiment Analysis

**Question:** "What about the fact that ≈47% of the tweets are about Tesla?"

**Analysis Added:** Stock-specific sentiment distribution check

**Key Metrics:**
1. **Coefficient of Variation (CV)** of sentiment across stocks
   - CV < 10%: LOW variation → sample weights OK
   - CV 10-20%: MODERATE → use weights cautiously
   - CV > 20%: HIGH → downsample recommended

2. **Extreme ticker identification**
   - Stocks with >±15% deviation from mean sentiment
   - Indicates stock-specific bias risk

**Purpose:** Determine if different stocks have different sentiment distributions
- If AAPL = 90% positive, TSLA = 65% positive, PG = 40% positive
- Then sample weights could amplify stock-specific shortcuts!

---

## 3. BALANCING STRATEGIES

### 3.1 Stratified Split vs Balanced Data

**Question:** "Explain stratified split more please. How does it handle Tesla majority?"

**IMPORTANT CLARIFICATION:**

**Stratified split does NOT balance data!**
- It maintains SAME proportions in train and test
- Both sets still have 47% Tesla
- Does NOT solve Tesla dominance

**What it does:**
```
Total: 47% TSLA, 12% TSM, 6.5% AAPL
↓ Stratified split
Train (80%): 47% TSLA, 12% TSM, 6.5% AAPL
Test (20%): 47% TSLA, 12% TSM, 6.5% AAPL
```

**Benefit:** Reliable evaluation (test set mirrors real distribution)

**What actually handles Tesla dominance:** SAMPLE WEIGHTS

---

### 3.2 Sample Weights - THE REAL SOLUTION

**How it works:**

**Without weights:**
- Tesla tweet: contributes 1.0 to loss
- PG tweet: contributes 1.0 to loss
- Total Tesla influence: 29,938 × 1.0 = 29,938
- Total PG influence: 515 × 1.0 = 515
- **Result:** Model focuses 58x more on Tesla!

**With balanced weights:**
```python
weight = total_samples / (n_classes × n_samples_in_class)

Tesla: 63,497 / (25 × 29,938) ≈ 0.085
PG: 63,497 / (25 × 515) ≈ 4.93
```

- Tesla tweet: contributes 0.085 to loss
- PG tweet: contributes 4.93 to loss
- Total Tesla influence: 29,938 × 0.085 ≈ 2,545
- Total PG influence: 515 × 4.93 ≈ 2,539
- **Result:** Equal influence! 🎯

**Analogy:**
- Classroom: Give quiet students microphones
- Each stock gets equal "voice" regardless of sample count

---

### 3.3 Advanced Concern: Stock-Specific Sentiment Bias

**Question:** "Imagine a stock with high sample weight has very imbalanced sentiment? E.g., AAPL has 90% positive sentiment - that's not good either, right?"

**Answer:** EXCELLENT catch - This is a real risk!

**The Problem:**
- If AAPL: 90% positive (with high weight from small sample size)
- Model might learn: "iPhone mentions → predict positive"
- Overfits to stock-specific patterns
- Fails on negative AAPL tweets or similar vocabulary elsewhere

**When it matters:**
1. **If stock name is a feature** - model can take shortcut
2. **If stocks have very different sentiment distributions** - CV > 20%
3. **High-weight + extreme sentiment** - amplifies bias

**Solution hierarchy:**
1. **Check CV first** (that's why we added the analysis!)
2. **If CV < 20%**: Use sample weights + text-only features
3. **If CV > 20%**: Downsample instead of weights
4. **Always evaluate per-stock** separately

**This is why the sentiment variation analysis is CRITICAL!**

---

### 3.4 Clarification: Why Downsample If Training Uses Different Data?

**Question:** "I will train my sentiment analysis model on the sentiment labeled dataset right? So it recognizes general sentiment. Then use that to predict the sentiment of the stocks. Why did I need to downsample the stocks to do that?"

**Answer:** You're ABSOLUTELY CORRECT about training - but downsampling serves a different purpose!

**Key Understanding:**

**Training happens on:** `labeled_sentiment_tweets.csv` (5,791 tweets)
- Has sentiment labels (1, -1)
- NO Tesla dominance issue
- Model learns general sentiment patterns
- **Downsampling does NOT affect this!** ✅

**Downsampling applies to:** `balanced_tweets_stock_data.csv` (for prediction & analysis)
- NO labels (these are unlabeled tweets)
- Has stock prices + returns
- Used AFTER model is trained

**Why Downsampling Matters (3 Reasons):**

**Reason 1: Fair Evaluation Metrics**
```python
# Without downsampling (47% Tesla):
predictions = model.predict(all_stock_tweets)
overall_f1 = f1_score(y_true, predictions)
# ⚠️ Dominated by Tesla performance!
# If F1=0.90 on Tesla, F1=0.60 on others
# Overall might show 0.80 but misleading

# With downsampling (balanced):
predictions = model.predict(balanced_stock_tweets)
overall_f1 = f1_score(y_true, predictions)
# ✅ All stocks contribute equally to metric
```

**Reason 2: Stock Return Correlation Analysis** (THE BIG ONE!)
```python
# Your ultimate goal: Does sentiment predict returns?

# Without downsampling:
daily_sentiment = stock_data.groupby('date')['predicted_sentiment'].mean()
daily_returns = stock_data.groupby('date')['daily_return'].mean()
correlation = daily_sentiment.corr(daily_returns)
# ⚠️ Really measuring: "Does sentiment predict TESLA returns?"

# With downsampling:
correlation = balanced_data...
# ✅ Measures: "Does sentiment predict returns ACROSS ALL STOCKS?"
```

**Reason 3: Per-Stock Evaluation Comparability**
- Tesla: 29,938 tweets → very reliable performance estimate
- Small stock: 515 tweets → unreliable estimate
- Can't fairly compare per-stock performance
- Balanced: All ~5,000 tweets → comparable sample sizes

**Data Flow:**
```
labeled_sentiment_tweets.csv (5,791 tweets, HAS labels)
    ↓ [TRAINING]
Trained Model
    ↓ [PREDICTION]
balanced_tweets_stock_data.csv (36,033 tweets, NO labels)
    ↓ [ANALYSIS]
Sentiment → Stock Return Correlations
```

**Bottom Line:**
- Training: Uses labeled data (not affected by downsampling)
- Prediction: Can work on any data
- Analysis & Evaluation: Needs balanced data for fair assessment

**The downsampling ensures your EVALUATION and CORRELATION ANALYSIS are fair across all stocks, even though it doesn't affect TRAINING at all.**

---

### 3.5 FinBERT Pre-trained Model Consideration

**Question:** "I want to use FinBERT, which is a sentiment model for financial data. But that is already pre-trained? Can't I use it directly on the balanced dataset?"

**Answer:** YES - You can use FinBERT directly!

**What FinBERT is:**
- Pre-trained on financial news, earnings calls, formal financial documents
- Already fine-tuned for financial sentiment
- Can be applied directly without training

**Follow-up Question:** "Part of the idea was to fine-tune FinBERT on labeled stock sentiments, but it is already fine-tuned on that?"

**Important Distinction:**

FinBERT IS trained on:
- ✅ Financial news articles
- ✅ Earnings call transcripts  
- ✅ Formal financial documents

FinBERT is NOT trained on:
- ❌ Twitter/social media language
- ❌ Informal financial discussions
- ❌ Slang, memes, emojis ("to the moon 🚀", "diamond hands")

**Your Project Value:**

Fine-tuning FinBERT on Twitter data still has merit:
1. **Domain gap**: Financial Twitter ≠ Financial news
   - Twitter: "TSLA to the moon! 🚀"
   - News: "Tesla shares appreciated 5%"

2. **Research contribution**: Test if FinBERT generalizes to social media

3. **Potential improvement**: Fine-tune on your labeled Twitter data

**Recommended Approach:**
```
1. Baseline: Test FinBERT out-of-box on labeled_sentiment_tweets.csv
2. Evaluate: Check accuracy on Twitter data
3. If accuracy > 85%: Use directly
4. If accuracy < 75%: Fine-tune on your labeled data
5. Compare: Show improvement from fine-tuning
```

**Project is still valid:** You're adapting financial NLP to social media domain!

**Use of labeled data with FinBERT:**
- Optional for evaluation (test FinBERT's Twitter performance)
- Optional for fine-tuning (if needed)
- Not required if FinBERT performs well as-is

---

## 4. BALANCING STRATEGY DECISION TREE

### Complete Strategy:

```
1. Run EDA Stock-Specific Sentiment Analysis
   ↓
2. Check Coefficient of Variation (CV)
   ↓
3. If CV < 10%:
   → Use stratified split + sample weights
   → Text-only model (no stock name feature)
   → Evaluate per-stock performance
   
   If CV 10-20%:
   → Use sample weights + careful monitoring
   → Text-only model
   → MUST check per-stock F1-scores
   → If some stocks fail, switch to downsampling
   
   If CV > 20%:
   → Downsample TSLA to 5-7k tweets
   → Keep other stocks as-is
   → Simpler, more robust
   → Less risk of stock-specific bias
```

**Key Principle:**
- Sample weights = handles CLASS imbalance (# of samples)
- Does NOT fix PATTERN differences (if TSLA language ≠ AAPL language)
- If language patterns differ by stock → Downsample is safer!

---

## 5. DATA AUGMENTATION CONSIDERATIONS

**Context:** Downsampling loses ~30% of data (25k Tesla tweets)

**Question:** "Could we potentially augment data?"

**Analysis:**

### Safe Augmentation Techniques:
1. **Back-translation** (English → French → English)
   - Preserves meaning and numbers
   - Natural variations
   - BUT: Slow, requires models

2. **Synonym replacement** (carefully)
   - Only non-financial words
   - Avoid: "calls", "puts", "buy", "sell", etc.

3. **Paraphrasing with T5/GPT**
   - High quality
   - Preserves sentiment
   - BUT: Requires API/compute

### Dangerous for Financial Data:
❌ Number replacement - changes sentiment completely
❌ Ticker replacement - changes context
❌ Random word operations - creates unnatural sentences

### Recommendation:
**DON'T augment - here's why:**

1. **You have enough data:** 60k+ tweets is plenty for sentiment
2. **Quality > quantity:** Real data always better than synthetic
3. **Financial domain risk:** Easy to change meaning subtly
4. **Better alternatives exist:**
   - Compromise: Downsample TSLA to 10-15k (not 5k)
   - Use sample weights + regularization
   - Focus on robust evaluation strategy

**If you must augment:**
- Only stocks with < 500 tweets
- Use back-translation
- Manually validate quality
- Still use sample weights

**Decision:** Prefer downsampling + weights over augmentation.

---

## 6. FEATURE ENGINEERING (Future)

### Decision: Wait Until After Baseline

**Question:** "I don't know if it's a good idea to add [features like TFSI] now already?"

**Answer:** CORRECT - Don't engineer features yet!

**Reasoning:**
- Don't know what works until you have baseline
- Risk of premature optimization
- Feature engineering should be guided by model weaknesses
- Analysis paralysis with too many features

**Proper workflow:**
```
1. Build baseline model ← Start here
2. Evaluate performance
3. Identify weaknesses
4. Engineer targeted features ← Now you know what to add
5. Iterate
```

**Example:**
- If: "Sentiment alone doesn't predict well"
- Then: Add volume features
- If: "Same-day correlation weak"
- Then: Add lag features (next-day returns)

**Features to consider later:**
- TFSI (Twitter Financial Sentiment Index)
- Tweet volume per day
- Sentiment dispersion/volatility
- Rolling sentiment windows
- Mood indicators

**Decision:** Hold on features until baseline reveals gaps.

---

## 7. TIME-BASED ANALYSIS CONSIDERATION

**Question:** "What if I wanted to check if specific tweet at certain timestamp made stock go down?"

**Current limitation:** Only daily stock data, not intraday

**Your data:**
- Tweets: Precise timestamps (2022-09-29 23:41:16)
- Stock prices: Daily only (end-of-day close)

**Can't do:**
- "Did 2PM tweet cause 3PM price drop?"
- Requires minute/hourly price data

**CAN do:**
- "Do today's tweets predict tomorrow's movement?"
- Daily sentiment aggregation vs next-day returns
- Correlation analysis

**Solution for timestamp analysis:**
Would need:
1. Intraday stock data (minute-level)
2. Market hours filtering
3. Lead/lag analysis (T+15min, T+30min)

**For your project:** Focus on daily prediction
- Aggregate daily sentiment
- Predict next-day returns
- This is standard in academic research

---

## 8. KEY TAKEAWAYS & DECISIONS

### Critical Decisions Made:

1. **Text Preprocessing:**
   - ✅ Keep digits and financial symbols ($, %)
   - ✅ Preserve tickers with "ticker_" prefix
   - ✅ Keep both original and cleaned columns

2. **Sentiment Imbalance (1.75:1):**
   - ✅ Acceptable - no immediate action
   - ✅ Use F1-score for evaluation
   - ✅ Monitor per-class performance

3. **Stock Imbalance (47% TSLA):**
   - 🚨 CRITICAL ISSUE - must address
   - ✅ Strategy depends on CV from EDA
   - ✅ Likely: Stratified split + sample weights OR downsampling

4. **Feature Engineering:**
   - ⏸️ Wait until after baseline model
   - ✅ Let data reveal what features to add

5. **Data Augmentation:**
   - ❌ Not recommended for this project
   - ✅ Have sufficient data (60k+ tweets)
   - ✅ Quality > quantity in financial NLP

### Next Steps:

1. **Complete EDA** - run stock-specific sentiment analysis
2. **Check CV value** - determines balancing strategy
3. **Implement chosen strategy**:
   - If CV < 20%: Sample weights
   - If CV > 20%: Downsample TSLA
4. **Build baseline sentiment classifier**
5. **Evaluate per-stock performance**
6. **Iterate based on results**

### Success Metrics:

**Overall:**
- F1-score > 0.70 (both classes)
- Balanced precision and recall

**Per-stock:**
- All stocks: F1 > 0.60
- No stock: F1 < 0.50
- TSLA performance ≠ only good performance

**Model should work across stocks, not just Tesla!**

---

## 9. TECHNICAL NOTES

### Important Functions:

```python
# Daily returns with proper grouping
stock_prices['daily_return'] = stock_prices.groupby('Stock Name')['Close'].pct_change()

# Text cleaning with ticker preservation
text = re.sub(r'\b([A-Z]{2,5})\b', r'ticker_\1', text)
text = text.lower()

# Stratified split
X_train, X_test = train_test_split(X, y, stratify=stock_name, random_state=42)

# Sample weights
from sklearn.utils.class_weight import compute_sample_weight
weights = compute_sample_weight('balanced', stock_train)
model.fit(X_train, y_train, sample_weight=weights)
```

### Evaluation Template:

```python
# Overall metrics
print(classification_report(y_test, y_pred))

# Per-stock evaluation
for stock in test_stocks.unique():
    mask = test_stocks == stock
    f1 = f1_score(y_test[mask], y_pred[mask])
    print(f"{stock}: F1={f1:.3f}")
```

---

## 10. PROJECT METADATA

**Dataset Statistics:**
- Labeled sentiment tweets: 5,791 (training)
- Stock tweets with prices: 63,497 (prediction)
- Stocks covered: 25
- Date range: Oct 2021 - Sep 2022 (1 year)
- Most tweeted: TSLA (47%), TSM (12%), AAPL (6.5%)

**Files:**
- `EEX.ipynb` - Data processing pipeline
- `02_EDA.ipynb` - Exploratory analysis + balancing recommendations
- `data/processed/` - Cleaned and merged datasets
- `PROJECT_NOTES.md` - This file (decisions and rationale)

---

## 11. ORGANIZATIONAL DECISIONS

### File Structure

**Question:** "Should I keep going in the notebook structure or create a new file named EDA?"

**Answer:** Create separate notebooks!

**Recommended structure:**
```
AI-financial-sentiment-prediction/
├── 01_data_processing.ipynb (or EEX.ipynb)
├── 02_EDA.ipynb
├── 03_sentiment_model.ipynb (future)
├── 04_prediction_model.ipynb (future)
├── data/
│   ├── raw/
│   └── processed/
├── PROJECT_NOTES.md
└── requirements.txt
```

**Benefits:**
- ✅ Modular and clean
- ✅ Fast iteration (reload CSVs vs rerun everything)
- ✅ Easy to navigate
- ✅ Professional structure
- ✅ Each notebook has one purpose

**Decision:** Separate notebooks for each stage of pipeline.

---

**Last Updated:** October 12, 2025  
**Status:** EDA Complete, Ready for Model Building  
**Next:** Sentiment classifier training with chosen balancing strategy

---

*This document captures all key decisions, questions, and rationale throughout the project. Use it for your project report, presentations, and to remember why specific choices were made.*

---

### 3.6 Handling FinBERT's 3-Class Output with Binary Data

**Question:** "How would I map my binary sentiment data to FinBERT's outputted pos, neg, neutral? Do I skip the neutral examples entirely?"

**Context:** FinBERT predicts 3 classes (positive, neutral, negative) but our training data only has 2 classes (positive, negative).

#### Two Approaches Considered:

**Option 1: Reshape Model to 2 Classes**
- Force FinBERT to output only positive/negative
- Requires reshaping the classification head from 3 to 2 outputs
- Problem: Neutral concept still exists in BERT encoder but no output for it
- Result: Neutral tweets get randomly classified as pos/neg with low confidence
- Example: "no change today" might predict positive at 52% (barely)

**Option 2: Keep 3 Classes, Handle Neutrals During Prediction**
- Maintain FinBERT's original 3-class structure
- Model can output neutral predictions
- We decide how to handle neutrals in post-processing
- Problem: Neutral class never gets training examples from our data
- Solution: Filter out neutral predictions or map them strategically

#### Decision Made: Option 2 - Reshape Model to 2 Classes

**Rationale:**
1. The model wont be able to predict any neutral during train/val either way so it will simple train its way to a 2 class output. Leading to unnecessary parameters and training time. 

---