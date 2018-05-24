from flask import render_template, flash, redirect, url_for, request
from flask_login import login_user, logout_user, current_user, login_required
from werkzeug.urls import url_parse
from app import app, db
from app.forms import LoginForm, RegistrationForm, EditProfileForm, AddFlashcardForm
from app.models import User, flashcard
from datetime import datetime
import os
import random
import pandas as pd
import numpy as np

# class processing():
import json
import urllib.request
import pandas as pd
import pickle

# Gensim imports
from gensim import corpora
from gensim.corpora import Dictionary, MmCorpus

# Pull in dictionary
vocabulary = corpora.Dictionary.load('app/cards.vocab')


@app.before_request
def before_request():
    if current_user.is_authenticated:
        current_user.last_seen = datetime.utcnow()
        db.session.commit()

@app.route('/')
@app.route('/index')
@login_required
def index():

    return render_template('index.html', title='Home')


@app.route('/login', methods=['GET', 'POST'])
@app.route('/login/', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password')
            return redirect(url_for('login'))
        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get('next')
        if not next_page or url_parse(next_page).netloc != '':
            next_page = url_for('index')
        return redirect(url_for('index'))
    return render_template('login.html', title='Sign In', form=form)

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('You are now registered')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)


@app.route('/user/<username>')
@login_required
def user(username):
    user = User.query.filter_by(username=username).first_or_404()
    posts = [
        {'author': user, 'body': 'Topic modeling can likely find underlying patterns that educators might miss.'},
        {'author': user, 'body': 'Spaced repetition reinforces memory by leveraging the utility of forgetting.'}
    ]
    return render_template('user.html', user=user, posts=posts)

@app.route('/edit_profile', methods=['GET', 'POST'])
@login_required
def edit_profile():
    form = EditProfileForm(current_user.username)
    if form.validate_on_submit():
        current_user.username = form.username.data
        current_user.about_me = form.about_me.data
        db.session.commit()
        flash('Your changes have been saved.')
        return redirect(url_for('edit_profile'))
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.about_me.data = current_user.about_me
    return render_template('edit_profile.html', title='Edit Profile', form=form)

@app.route('/follow/<username>')
@login_required
def follow(username):
    user = User.query.filter_by(username=username).first()
    if user is None:
        flash('User {} not found.'.format(username))
        return redirect(url_for('index'))
    if user == current_user:
        flash('You cannot follow yourself!')
        return redirect(url_for('user', username=username))
    current_user.follow(user)
    db.session.commit()
    flash('You are following {}!'.format(username))
    return redirect(url_for('user', username=username))

@app.route('/unfollow/<username>')
@login_required
def unfollow(username):
    user = User.query.filter_by(username=username).first()
    if user is None:
        flash('User {} not found.'.format(username))
        return redirect(url_for('index'))
    if user == current_user:
        flash('You cannot unfollow yourself!')
        return redirect(url_for('user', username=username))
    current_user.unfollow(user)
    db.session.commit()
    flash('You are not following {}.'.format(username))
    return redirect(url_for('user', username=username))

@app.route('/about')
@app.route('/about/')
def about():
    user = {'username': 'Tovio'}
    return render_template('about.html', title='about', user=user)



@app.route('/add_card', methods=['GET', 'POST'])
def add_card():
    form = AddFlashcardForm()
    if form.validate_on_submit():
        card = flashcard(front=form.front.data, back=form.back.data)
        db.session.add(card)
        db.session.commit()
        flash('Card Added!')
        return redirect(url_for('add_card'))
    return render_template('add_card.html', title='Register', form=form)

@app.route('/add_corpus_card', methods=['GET','POST'])
def add_corpus_card():
    file_path = 'app/history_flashcards.txt'

    df = pd.read_csv(file_path, sep='\t', names=['question','answer'])

    n_samples = 10

    rows = np.random.choice(df.index.values, n_samples)
    choice = random.choice(rows)
    row = df.iloc[choice]

    front = row['question']
    back = row['answer']

    print('\n\n\n\n***********ADDING*******************')
    print('row#: {}'.format(rows))
    print('question: {}'.format(row['question']))
    print('answer: {}'.format(row['answer']))
    print('************************************\n\n\n\n')

    card = flashcard(front=front, back=back)
    db.session.add(card)
    db.session.commit()
    flash('Similar Card Added!')

    return redirect(url_for('review_cards'))

@app.route('/review_cards', methods=['GET', 'POST'])
@login_required
def review_cards():

    card = flashcard.query.order_by(flashcard.timestamp.asc()).first()
    # card = random.choice(card)

    # update the timestamp
    card.timestamp = datetime.utcnow()
    db.session.commit()

    return render_template('review_cards.html', title='Register', card=card)

@app.route('/increase_score', methods=['GET','POST'])
def increase_score():
    card = flashcard.query.order_by(flashcard.timestamp.desc()).first()
    card.score += 1
    db.session.commit()
    return redirect(url_for('review_cards'))

@app.route('/decrease_score', methods=['GET','POST'])
def decrease_score():
    card = flashcard.query.order_by(flashcard.timestamp.desc()).first()
    card.score -= 1
    db.session.commit()
    return redirect(url_for('review_cards'))
