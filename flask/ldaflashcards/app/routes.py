from flask import render_template, flash, redirect, url_for, request
from flask_login import login_user, logout_user, current_user, login_required
from werkzeug.urls import url_parse
from app import app, db
from app.forms import LoginForm, RegistrationForm, EditProfileForm, AddFlashcardForm
from app.models import User, flashcard
from datetime import datetime
import os

# class processing():
import json
import urllib.request
import pandas as pd
import pickle

# class preparedata(object):
#
#     def apply(self):
#         self.get_request()
#         self.cleanit()
#         return self.new_df
#
#     def get_request(self):
#
#         url = 'http://galvanize-case-study-on-fraud.herokuapp.com/data_point'
#         respons = urllib.request.urlopen(url)
#         data = json.loads(respons.read().decode(respons.info().get_param('charset') or 'utf-8'))
#         new_dict = {}
#         new_dict1 = {}
#         cols = ['body_length', 'channels', 'delivery_method', 'fb_published', 'gts',
#        'has_analytics', 'has_header', 'has_logo', 'name_length', 'num_order', 'num_payouts',
#        'object_id', 'org_facebook', 'org_twitter', 'sale_duration',
#        'sale_duration2', 'show_map', 'user_age', 'user_type', 'venue_latitude',
#        'venue_longitude', 'has_xml', 'has_org_descr', 'count_event_id',
#        'total_av', 'profit', 'quant_sold', 'quant_total', 'spam_domain',
#        'number_payouts', 'dif_address', 'dif_names', 'total_amount',
#        'mean_amount']
#         new_dict['description'] = ['cat']
#         for c in cols:
#             new_dict[c] = [0]
#         for k, v in data.items():
#             new_dict1[k] = [v]
#         new_dict1.update(new_dict)
#         self.df = pd.DataFrame.from_dict(new_dict1)
#
#
#     def cleanit(self):
#         date_time_list = ('approx_payout_date', 'event_created', 'user_created', 'event_end', 'event_published', 'event_start')
#         self.cleaned_df = clean(self.df, date_time_list)
#         self.new_df = self.cleaned_df.apply_functions(LDA=False)
#
#     def predict_proba(self, model, X):
#         """Make probability predictions on new data.
#         Parameters
#         ----------
#         X: A numpy array or list of text fragments, to be used as predictors.
#         Returns
#         -------
#         probs: A (n_obs, n_classes) numpy array of predicted class probabilities.
#         """
#
#         return model.predict_proba(X)


@app.before_request
def before_request():
    if current_user.is_authenticated:
        current_user.last_seen = datetime.utcnow()
        db.session.commit()

@app.route('/')
@app.route('/index')
@login_required
def index():
    posts = [
        {
            'author': {'username':'John'},
            'body': 'Beautiful day in Portland!'

        },
        {
            'author': {'username':'Brobb'},
            'body': 'Beautiful day in Skoortland!'
        }

    ]

    return render_template('index.html', title='Home', posts=posts)


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


# @app.route('/predict')
# def b(name=None):
#
#     p = preparedata()
#     cleaned_df = p.apply()
#     path = 'app/models/finalized_model.pkl'
#     # with open(path) as f:
#     model = pickle.load(open('app/models/finalized_model.pkl', 'rb'))
#     print(cleaned_df.columns)
#     # pred1 = str(model.predict_proba([data])[0])
#     pred = p.predict_proba(model, cleaned_df)
#
#     print('Likelihood that this is fraud: {0:0.2%}'.format(pred[0][1]))
#     # new_df = cleaned_df.apply_functions(LDA=False)
#     timestamp = datetime.utcnow()
#     data = cleaned_df.to_string()
#
#     prediction = fraudprediction(timestamp=timestamp, data=data, prediction=pred[0][1])
#     db.session.add(prediction)
#     db.session.commit()
#
#     return 'Likelihood that this is fraud: {0:0.2%}'.format(pred[0][1])

@app.route('/user/<username>')
@login_required
def user(username):
    user = User.query.filter_by(username=username).first_or_404()
    posts = [
        {'author': user, 'body': 'Topic modeling can likely find underlying patterns in a corpus that educators might miss.'},
        {'author': user, 'body': 'Spaced repetition reinforces memory by leveraging the utility of forgetting'}
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

@app.route('/review_cards')
@login_required
def review_cards():
    # TODO: Exclude scores greater than 5
    card = flashcard.query.order_by(flashcard.timestamp.asc()).first()
    print(card)

    # update the timestamp
    card.timestamp = datetime.utcnow()
    db.session.commit()

    # TODO: If no cards with score less than 6, add new card

    # TODO: If timestamp is < 3 minutes from now, add new card

    return render_template('review_cards.html', card=card)
