# PROJECT: AI-Based Fake News Detection System
# What This File Will Do;
#   1. Loads Fake.csv and True.csv from the dataset/ folder
#   2. Cleans the text using NLP techniques
#   3. Converts text to numbers using TF-IDF
#   4. Trains Logistic Regression and Naive Bayes models
#   5. Prints accuracy results in the terminal
#   6. Saves 4 plots as PNG images in the plots/ folder
#   7. Saves trained models in the models/ folder


#Import Libraries
import pandas as pd  # Load and manage CSV data
import numpy as np   # Numerical operations
import re            # Text pattern matching (regex)
import pickle        # Save/load Python objects
import os            # Create folders
import nltk
from nltk.corpus import stopwords         # Common English words to remove
from nltk.stem import WordNetLemmatizer  # Reduce words to root form
from sklearn.model_selection import train_test_split       # Split data into train/test
from sklearn.feature_extraction.text import TfidfVectorizer # Convert text to numbers
from sklearn.linear_model import LogisticRegression         # ML Model 1
from sklearn.naive_bayes import MultinomialNB               # ML Model 2
from sklearn.metrics import (
    accuracy_score,        # % correct predictions
    precision_score,       # Of predicted Fake, how many are actually Fake?
    recall_score,          # Of all Fake news, how many did we catch?
    f1_score,              # Balance between precision and recall
    confusion_matrix,      # Table of TP, TN, FP, FN
    classification_report  # Full metrics table
)
import matplotlib.pyplot as plt  # Create charts
import seaborn as sns   # Better looking charts
from wordcloud import WordCloud  # Word cloud images

# STEP 1: Download NLTK Data Files
# These are downloaded once and saved on your computer
print("=" * 55)
print("  FAKE NEWS DETECTION — MODEL TRAINING STARTED")
print("=" * 55)
print("\n[1/8] Downloading NLTK data...")
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('omw-1.4',   quiet=True)
print("      Done!")

# STEP 2: Load Dataset
# Fake.csv = fake news   → we label these as 1
# True.csv = real news   → we label these as 0
print("\n[2/8] Loading dataset...")

try:
    fake_df = pd.read_csv("dataset/Fake.csv")  # Load fake news file
    true_df = pd.read_csv("dataset/True.csv")  # Load real news file
except FileNotFoundError:
     print("\nERROR: Dataset files not found!")
     print("Run: python create_sample_dataset.py")
     print("OR download from Kaggle:")
     print("https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
     exit()

fake_df["label"] = 1  # 1 means Fake
true_df["label"] = 0  # 0 means Real

# Combine both into one big table
df = pd.concat([fake_df, true_df], ignore_index=True)
# Use both title AND body text (more text = better predictions)
df["content"] = df["title"] + " " + df["text"]
# Shuffle rows so Fake and Real are mixed together
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Total articles :{len(df)}")
print(f"Fake news :{df['label'].sum()}")
print(f"Real news :{(df['label']==0).sum()}")

# STEP 3:Clean Text(NLP Pre-Processing)
# Raw text is messy. We must clean it before ML can use it.
print("\n[3/8] Cleaning text (NLP preprocessing)...")

lemmatizer = WordNetLemmatizer()  # Reduces words to root form
stop_words  = set(stopwords.words('english')) # Words like "the", "is", "a"

def clean_text(text):
      """
    Cleans one news article through the NLP pipeline.
    Called on every single article in the dataset.
    """
      
      text = re.sub(r'http\S+|www\S+', '', str(text)) # Remove URLs
      text = re.sub(r'[^a-zA-Z\s]', '', text)     # Remove numbers and symbols
      text = text.lower()      # Make all letters lowercase
      words = text.split()     # Split words
      words = [
        lemmatizer.lemmatize(w)  # Convert to root: "running" → "run"
        for w in words
        if w not in stop_words  # Skip "the", "is", "and" etc.
        and len(w) > 2      # Skip single letters and 2-letter words
    ]
      return " ".join(words)    #Join Words Back

# Apply clean_text() to every article in the dataset
df["cleaned"] = df["content"].apply(clean_text)
print("Text cleaning complete!")

# STEP 4: Convert Text to Numbers (TF-IDF)
# ML models only understand numbers, not words.
# TF-IDF gives each word in each article an importance score.
# Word that appears a lot in ONE article but rarely in others → high score
# Word that appears in EVERY article → low score (not useful)
print("\n[4/8] Converting text to numbers (TF-IDF)...")

X = df["cleaned"]  # Input  = cleaned article text
y = df["label"]    # Output = 0 (Real) or 1 (Fake)

# Split: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% = test set
    random_state=42,    # Same split every run
    stratify=y          # Equal fake/real ratio in both sets
)

# Create TF-IDF vectorizer
# max_features = only use top 50000 most important words
# ngram_range=(1,2) = consider single words AND pairs of words
tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))

# fit_transform = learn vocabulary from training data + convert to numbers
X_train_vec = tfidf.fit_transform(X_train)

# transform only = convert test data using SAME vocabulary (never fit on test data!)
X_test_vec  = tfidf.transform(X_test)

print(f"Training samples: {X_train_vec.shape[0]}")
print(f"Testing  samples: {X_test_vec.shape[0]}")

# STEP 5: Train the ML Models
# Feed the numerical data into our models so they can learn patterns.
# Training = the model figures out which word patterns = Fake vs Real.
print("\n[5/8] Training ML models...")

# Model 1: Logistic Regression
# Finds a mathematical boundary separating Fake and Real news
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_vec, y_train) # Train on training data
lr_predictions = lr_model.predict(X_test_vec)  # Predict on test data
print("Logistic Regression: Done!")

# Model 2: Naive Bayes
# Calculates probability: P(Fake|these words) vs P(Real|these words)
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
nb_predictions = nb_model.predict(X_test_vec)
print("Naive Bayes: Done!")

# STEP 6: Evaluate How Well Models Performed
# Compare predictions vs actual correct answers
print("\n[6/8] Evaluating models...")

def show_results(name, y_true, y_pred):
     """Prints all performance metrics for a model."""
     acc  = accuracy_score(y_true, y_pred)
     prec = precision_score(y_true, y_pred, zero_division=0)
     rec  = recall_score(y_true, y_pred, zero_division=0)
     f1   = f1_score(y_true, y_pred, zero_division=0)

     print(f"\n{'─'*40}")
     print(f"{name}")
     print(f"{'─'*40}")
     print(f"Accuracy  : {acc*100:.2f}%  ← Overall correct predictions")
     print(f"Precision : {prec*100:.2f}%  ← When predicted Fake, how often correct?")
     print(f"Recall    : {rec*100:.2f}%  ← Of all Fake news, how many caught?")
     print(f"F1-Score  : {f1*100:.2f}%  ← Balance of Precision and Recall")
     print(f"\n{classification_report(y_true, y_pred, target_names=['Real','Fake'], zero_division=0)}")
     return acc, prec, rec, f1

lr_scores = show_results("Logistic Regression", y_test, lr_predictions)
nb_scores = show_results("Naive Bayes",  y_test, nb_predictions)

# STEP 7: Generate and Save Visualization Plots
# Creates 4 PNG image files saved in the plots/ folder
print("\n[7/8] Generating visualizations...")
os.makedirs("plots", exist_ok=True)

# ── Plot 1: Confusion Matrices ──
# Shows correct vs wrong predictions for each model
# Rows = what the article ACTUALLY was (Real/Fake)
# Columns = what the model PREDICTED (Real/Fake)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Confusion Matrices", fontsize=16, fontweight='bold')
for ax, preds, title, cmap in zip(
    axes,
    [lr_predictions, nb_predictions],
    ["Logistic Regression", "Naive Bayes"],
    ["Blues", "Greens"]
):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                xticklabels=["Real","Fake"], yticklabels=["Real","Fake"])
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("plots/confusion_matrices.png", dpi=150)
plt.close()
print("Saved: plots/confusion_matrices.png")

# ── Plot 2: Model Comparison Bar Chart ──
categories = ['Accuracy','Precision','Recall','F1-Score']
x = np.arange(len(categories)); w = 0.35
fig, ax = plt.subplots(figsize=(10,6))
b1 = ax.bar(x-w/2, [s*100 for s in lr_scores], w, label='Logistic Regression', color='#3498db', alpha=0.9)
b2 = ax.bar(x+w/2, [s*100 for s in nb_scores], w, label='Naive Bayes',         color='#2ecc71', alpha=0.9)
for b in list(b1)+list(b2):
    h = b.get_height()
    ax.text(b.get_x()+b.get_width()/2, h+0.3, f'{h:.1f}%', ha='center', fontsize=9)
ax.set_title("Model Performance Comparison", fontsize=15, fontweight='bold')
ax.set_ylabel("Score (%)"); ax.set_xticks(x); ax.set_xticklabels(categories)
ax.set_ylim(0, 110); ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("plots/model_comparison.png", dpi=150)
plt.close()
print(" Saved: plots/model_comparison.png")

# ── Plots 3 & 4: Word Clouds ──
# Larger word = appears more often in that category
for text, label, cmap in [
    (" ".join(df[df["label"]==1]["cleaned"].values), "fake", "Reds"),
    (" ".join(df[df["label"]==0]["cleaned"].values), "real", "Greens")
]:
    wc = WordCloud(width=900, height=400, background_color="white",
                   colormap=cmap, max_words=100, collocations=False).generate(text)
    plt.figure(figsize=(11,5))
    plt.imshow(wc, interpolation="bilinear"); plt.axis("off")
    plt.title(f"Common Words — {label.title()} News", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"plots/wordcloud_{label}.png", dpi=150)
    plt.close()
    print(f"Saved: plots/wordcloud_{label}.png")

# STEP 8: Save Trained Models to Disk
# Save models as .pkl files so the web app can load them instantly
# without needing to retrain every time
print("\n[8/8] Saving trained models...")
os.makedirs("models", exist_ok=True)

with open("models/lr_model.pkl","wb") as f: pickle.dump(lr_model, f)
with open("models/nb_model.pkl","wb") as f: pickle.dump(nb_model, f)
with open("models/tfidf.pkl",   "wb") as f: pickle.dump(tfidf,    f)

print("      Saved: models/lr_model.pkl")
print("      Saved: models/nb_model.pkl")
print("      Saved: models/tfidf.pkl")

print("\n" + "=" * 55)
print("  TRAINING COMPLETE!")
print("  Accuracy shown above in terminal.")
print("  Plots saved in the plots/ folder.")
print("  Now run: streamlit run app.py")
print("=" * 55)



