[Google Presentation](https://docs.google.com/presentation/d/1382PGj1Ljha43d8BnQAKAfSGJM_bGphilbakYj5520A/edit?usp=sharing)

# In Search of a Smarter Flashcard
## Discerning Topics, Pedagogical Implications
#### Tovio Roberts, Capstone 2 @Galvanize


### **GOALS:**
- Clean flash card pool in a way that can be generalized to new card content
- Topic Model in a ‘reasonable’ way so as to enable simple similarity selection in an application
- Discuss discerning strong and weak subjects for a user
  - How to deliver the most "helpful" study materials
  - Associative Database --> Extrapolative Learning

### **DATA**
3 collections of ~12,000 each, flash cards, ~36,000 total.  These are divided into 3 general categories:
- Data Science
- Biology
- History

#### Each "card" is composed of a question and an answer.

The data sets are compiled from three sources:
- AnkiWeb
- Quizlet
- My own collection

### **PROJECT PROGRESSION Minimum Viable Product:**
1. Create data cleaning pipeline.
    * Strip html from cards
    * Standardize, modify or ignore formulas that are not consistent across cards.
    * Modify entries that lead to erroneous topics.
2. Explore NLP strategies to allow for meaningful clustering
    * Stem, Lemmatize, Stopwords
    * Count Vector
    * TF-IDF Vector
3. Use Clustering to analyze topics within a single subject corpus.
    * Provide list of “quintessential” words for each topic, most-common words per category.
    * User chosen categories become target labels.
4. Apply same Topic modeling to the full pool of cards


### *Improvement 1: Provide a simple API for flashcards*
1. Build topic distribution table when new cards are added
2. Retrieve flashcard
3. Update success table for flashcard user

### *Improvement 2: Provide an Interface for Card Review*
1. Swipe Left/Swipe Right simple front end.
2. Update success/fail.
3. Discern “Strong” and “Weak” topics.

### *Improvement 3: Smart Flashcard Delivery*
1. Incorporate Spaced Repetition and randomness settings into reviews.
2. Use similarity metrics to discern “Weak” and “Strong” topics, based on card review successes.
3. Deliver review cards as a function of spaced repetition, strength, and similarity.
