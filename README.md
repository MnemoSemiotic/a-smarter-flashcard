[Google Presentation](https://docs.google.com/presentation/d/1382PGj1Ljha43d8BnQAKAfSGJM_bGphilbakYj5520A/edit?usp=sharing)

# In Search of a Smarter Flashcard
## Discerning Topics, Predicting Categories
#### Tovio Roberts, Capstone 2 @Galvanize


### *GOALS:*
- Clean and categorize flash cards in a ‘reasonable’ way through use of NLP
- Predict the category of new flash card entries
- Discern strong and weak subjects for a user.

### *DATA*
I have a collection of 10,000 flash cards, composed of questions and answers, compiled by students of data science, ml, and statistics. There are many more cards available, and, if need be, I can include other subjects. The data are collected from three sources:
- AnkiWeb
- Quizlet
- My own collection

#### A single entry is composed of a question and answer.

### *PROJECT PROGRESSION Minimum Viable Product:*
1. Create data cleaning pipeline.
Drop cards that are not viable (e.g., composed of images). - Standardize formulas that are not consistent across cards. - Modify entries that lead to erroneous topics.
2. Use Clustering to discern general topics.
Provide list of “quintessential” cards, most-common words per category. - User chosen categories become target labels.
3. Predict the category of a new card.

### *Improvement 1: Provide an Interface for Card Review*
1. Swipe Left/Swipe Right simple front end.
2. Update success/fail.
3. Discern “Strong” and “Weak” topics.

### *Improvement 2: Build Questions from Answers*
1. Generate flash cards from pasted blocks of text.

### *Improvement 3: Smart Flashcard Delivery*
1. Incorporate Spaced Repetition.
2. Balance mixture of “Weak” and “Strong” topics.
