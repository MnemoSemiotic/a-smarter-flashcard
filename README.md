

# In Search of a Smarter Flashcard
## Part 1: Discerning Topics, Pedagogical Implications
#### Tovio Roberts, Capstone 2 @Galvanize


### **GOALS:**
- Clean flash card pool in a way that can be generalized to new card content
- Topic Model so as to enable simple similarity selection in an application
- Discuss discerning strong and weak subjects for a user
  - How to deliver the most "helpful" study materials
  - Associative Database --> Extrapolative Learning
--------------------------------------------------------------------------------
## **Questions:**
1. How can we clean and cluster/discern topics in a pool of flash cards in a ‘reasonable’ way through use of NLP techniques?
2. How will clustering generalize to a new pool of flash cards.
3. What is an effective structure to discern strong and weak subjects for an individual who regularly reviews flash cards.
--------------------------------------------------------------------------------
## **Problem Context**
- Learning is non-linear, extrapolative and diverse
- Traditional Education tends to be linear, grade-focused and "forgetful"
- Discerning underlying patterns of topics in a corpus of educational materials (such as flashcards, assignments) has the potential to facilitate extrapolative study
--------------------------------------------------------------------------------
## **Inspirations**
- [Anki](http://ankisrs.net) - Spaced-Repetition flash card program
- [PiggyDB](https://piggydb.net/) - associative, tag-based database for study and extrapolation
- [KhanAcademy](http://khanacademy.org) - Exploratory, non-grade-based educational model
- [Moonwalking with Einstein](https://joshuafoer.com/) by Joshua Forrer - Memory Palaces/Method of Locii
--------------------------------------------------------------------------------
## **The DATA:**
3 collections of ~12,000 each, flash cards, ~36,000 total.  These are divided into 3 general categories:
- Data Science
- Biology
- History

#### Each "card" is composed of a question and an answer.

The data sets are compiled from three sources:
- AnkiWeb
- Quizlet
- My collection
--------------------------------------------------------------------------------
## **The Data:** 1 observation == 1 Flash Card
- Front:
  - Can be in the form of a question, an Imperative (e.g., “Describe the Bernoulli Distribution:”)
  - May contain the majority of information from the observation as in the case of giving a definition and requesting the corresponding term.
    - Important to consider when later attempting to train an NN to write questions corresponding to answers.

- Back:
  - Effectively the same as the front.

--------------------------------------------------------------------------------
## **The Data:** Cleaning
- Front and Back may contain images, formulas, inconsistent vocabulary
Off-topic cards are likely present
  - Formulas have variety of representations
  - Images must be stripped

- With that in mind, let's consider a data model...
--------------------------------------------------------------------------------
## **The Data:** Simple MVC
![Simple MVC Concept](images/07_simple_app_design.png)
--------------------------------------------------------------------------------
### **PROJECT PROGRESSION Minimum Viable Product:**
1. Create data cleaning pipeline.
    * Strip html from cards
    * Standardize, modify or ignore formulas that are not consistent across cards.
    * Modify entries that lead to erroneous topics.
2. Explore NLP strategies to allow for meaningful clustering
    * Stem, Lemmatize, Stopwords
    * Count Vector
    * TF-IDF Vector
3. Use Topic Modeling to analyze topics within a single subject corpus.
    * Determine if discerned topics are sensible
4. Apply same Topic modeling to the full pool of cards
    * Provide list of salient and relevant words for each topic.

--------------------------------------------------------------------------------
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

--------------------------------------------------------------------------------

## Create data cleaning pipeline.
    * Strip html from cards
    * Standardize, modify or ignore formulas that are not consistent across cards.
    * Modify entries that lead to erroneous topics.

--------------------------------------------------------------------------------
## Explore NLP strategies to allow for meaningful clustering
    * Stem, Lemmatize, Stopwords
    * Count Vector
    * TF-IDF Vector

--------------------------------------------------------------------------------
## Use Topic Modeling to analyze topics within a single subject corpus.
    * Determine if discerned topics are sensible

--------------------------------------------------------------------------------
## Apply same Topic modeling to the full pool of cards
    * Provide list of salient and relevant words for each topic.
