from app import app, db
from app.models import User, Post, flashcard

# To export environment variable
#   $ export FLASK_APP=ldaflashcards.py

# To turn on debug, which will reload the application on every save
#   $ export FLASK_DEBUG=1

# create a shell context that adds the database
#   instance and models to the shell session
@app.shell_context_processor
def make_shell_context():
    return {'db':db, 'User':User, 'Post':Post, 'flashcard':flashcard}
