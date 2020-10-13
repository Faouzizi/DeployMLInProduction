from flask_wtf import FlaskForm
from wtforms import TextAreaField, SubmitField

class ContactForm(FlaskForm):
    data = TextAreaField("data")
    submit = SubmitField("Send")
