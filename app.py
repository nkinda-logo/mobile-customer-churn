# Standard Library Imports
import os
import io
import uuid
import tempfile
from datetime import datetime, timedelta
import logging

# Flask and Extensions
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFProtect
from wtforms import StringField, PasswordField, SelectField, FloatField, IntegerField
from wtforms.validators import DataRequired, Length, EqualTo, NumberRange
from werkzeug.security import generate_password_hash, check_password_hash

# Data Processing and ML
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from utils.data_processing import preprocess_data
from utils.visualization import create_visualizations

# PDF Generation
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Configuration
import config

# Initialize Flask App
app = Flask(__name__)
app.config.from_object(config.Config)

# Configure Logging
logging.basicConfig(level=logging.INFO)
app.logger.handlers = logging.getLogger().handlers

# Initialize CSRF Protection
csrf = CSRFProtect(app)


    

# Database Setup
db_dir = os.path.join(os.path.dirname(__file__), 'database')
os.makedirs(db_dir, exist_ok=True)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', f'sqlite:///{os.path.join(db_dir, "app.db")}')
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Login Manager Setup
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    predictions = db.relationship('Prediction', backref='user', lazy=True)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    batch_id = db.Column(db.String(36), nullable=True)
    customer_id = db.Column(db.String(50), nullable=True)
    telecom_company = db.Column(db.String(50))
    region = db.Column(db.String(50))
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    contract_type = db.Column(db.String(50))
    contract_duration = db.Column(db.String(50))
    tenure_months = db.Column(db.Integer)
    monthly_charges = db.Column(db.Float)
    data_usage_gb = db.Column(db.Float)
    call_duration_minutes = db.Column(db.Integer)
    complaints_filed = db.Column(db.Integer)
    customer_support_calls = db.Column(db.Integer)
    payment_method = db.Column(db.String(50))
    internet_service = db.Column(db.String(50))
    additional_services = db.Column(db.String(50))
    discount_offer_used = db.Column(db.String(10))
    billing_issues_reported = db.Column(db.Integer)
    prediction = db.Column(db.String(10))
    probability = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    report_generated = db.Column(db.Boolean, default=False)
    __table_args__ = (
        db.Index('idx_user_timestamp', 'user_id', 'timestamp'),
    )

# Forms
class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[Length(min=4, max=25)])
    email = StringField('Email Address', validators=[Length(min=6, max=100)])
    password = PasswordField('Password', validators=[
        DataRequired(),
        EqualTo('confirm', message='Passwords must match')
    ])
    confirm = PasswordField('Repeat Password')

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])

class PredictionForm(FlaskForm):
    telecom_company = SelectField('Telecom Company', choices=[
        ('Airtel', 'Airtel'), ('Tigo', 'Tigo'), ('Vodacom', 'Vodacom'),
        ('Halotel', 'Halotel'), ('TTCL', 'TTCL'), ('Zantel', 'Zantel')
    ], validators=[DataRequired()])
    region = SelectField('Region', choices=[
        ('Arusha', 'Arusha'), ('Dar es Salaam', 'Dar es Salaam'), ('Dodoma', 'Dodoma'),
        ('Geita', 'Geita'), ('Iringa', 'Iringa'), ('Kagera', 'Kagera'),
        ('Katavi', 'Katavi'), ('Kigoma', 'Kigoma'), ('Kilimanjaro', 'Kilimanjaro'),
        ('Lindi', 'Lindi'), ('Manyara', 'Manyara'), ('Mara', 'Mara'),
        ('Mbeya', 'Mbeya'), ('Morogoro', 'Morogoro'), ('Mtwara', 'Mtwara'),
        ('Mwanza', 'Mwanza'), ('Njombe', 'Njombe'), ('Pemba North', 'Pemba North'),
        ('Pemba South', 'Pemba South'), ('Pwani', 'Pwani'), ('Rukwa', 'Rukwa'),
        ('Ruvuma', 'Ruvuma'), ('Shinyanga', 'Shinyanga'), ('Simiyu', 'Simiyu'),
        ('Singida', 'Singida'), ('Songwe', 'Songwe'), ('Tabora', 'Tabora'),
        ('Tanga', 'Tanga'), ('Unguja North', 'Unguja North'), ('Unguja South', 'Unguja South')
    ], validators=[DataRequired()])
    age = IntegerField('Age', validators=[NumberRange(min=18, max=110)])
    gender = SelectField('Gender', choices=[('Male', 'Male'), ('Female', 'Female')], validators=[DataRequired()])
    contract_type = SelectField('Contract Type', choices=[
        ('Prepaid', 'Prepaid'), ('Postpaid', 'Postpaid'), ('Hybrid', 'Hybrid')
    ], validators=[DataRequired()])
    contract_duration = SelectField('Contract Duration', choices=[
        ('1 Month', '1 Month'), ('3 Months', '3 Months'), ('6 Months', '6 Months'),
        ('12 Months', '12 Months'), ('24 Months', '24 Months')
    ], validators=[DataRequired()])
    tenure_months = IntegerField('Tenure (Months)', validators=[NumberRange(min=1, max=120)])
    monthly_charges = FloatField('Monthly Charges', validators=[NumberRange(min=0)])
    data_usage_gb = FloatField('Data Usage (GB)', validators=[NumberRange(min=0)])
    call_duration_minutes = IntegerField('Call Duration (Minutes)', validators=[NumberRange(min=0)])
    complaints_filed = IntegerField('Complaints Filed', validators=[NumberRange(min=0)])
    customer_support_calls = IntegerField('Customer Support Calls', validators=[NumberRange(min=0)])
    payment_method = SelectField('Payment Method', choices=[
        ('Credit Card', 'Credit Card'), ('Bank Transfer', 'Bank Transfer'),
        ('Mobile Money', 'Mobile Money'), ('Cash', 'Cash'), ('Voucher', 'Voucher'), ('Other', 'Other')
    ], validators=[DataRequired()])
    internet_service = SelectField('Internet Service', choices=[
        ('Mobile Data', 'Mobile Data'), ('Fiber', 'Fiber'), ('DSL', 'DSL'),
        ('WiMAX', 'WiMAX'), ('None', 'None')
    ], validators=[DataRequired()])
    additional_services = SelectField('Additional Services', choices=[
        ('Streaming', 'Streaming'), ('VPN', 'VPN'), ('Cloud Storage', 'Cloud Storage'),
        ('Gaming', 'Gaming'), ('None', 'None')
    ], validators=[DataRequired()])
    discount_offer_used = SelectField('Discount Offer Used', choices=[
        ('Yes', 'Yes'), ('No', 'No')
    ], validators=[DataRequired()])
    billing_issues_reported = IntegerField('Billing Issues Reported', validators=[NumberRange(min=0)])

# User Loader for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Model Loading
def load_model():
    """Load the ML model with robust error handling and fallback"""
    model_path = 'models/churn_model.pkl'
    backup_path = 'models/backup_churn_model.pkl'
    try:
        model = joblib.load(model_path)
        app.logger.info("Loaded primary churn prediction model")
        return model
    except Exception as e:
        app.logger.error(f"Failed to load primary model: {str(e)}")
        try:
            model = joblib.load(backup_path)
            app.logger.warning("Using backup churn prediction model")
            return model
        except Exception as e:
            app.logger.critical(f"Failed to load backup model: {str(e)}")
            raise RuntimeError("No working model available")

model = load_model()

# Database Initialization
def initialize_database():
    """Initialize database and create admin user if not exists"""
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(username='admin').first():
            admin = User(
                username='admin',
                email='admin@example.com',
                password_hash=generate_password_hash('admin123', method='pbkdf2:sha256', salt_length=16),
                is_admin=True
            )
            db.session.add(admin)
            db.session.commit()
            app.logger.info("Admin user created")

initialize_database()

# Jinja2 Global Context
app.jinja_env.globals.update(
    now=datetime.now,
    timedelta=timedelta
)

# Routes: Static Pages
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

# Routes: Authentication
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        try:
            if User.query.filter_by(username=form.username.data).first():
                flash('Username already exists', 'danger')
                return redirect(url_for('register'))
            if User.query.filter_by(email=form.email.data).first():
                flash('Email already exists', 'danger')
                return redirect(url_for('register'))
            hashed_password = generate_password_hash(
                form.password.data, method='pbkdf2:sha256', salt_length=16
            )
            new_user = User(
                username=form.username.data,
                email=form.email.data,
                password_hash=hashed_password
            )
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Registration error: {str(e)}")
            flash('Registration failed. Please try again.', 'danger')
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if not user:
            flash('Username not found', 'danger')
            return redirect(url_for('login'))
        if not check_password_hash(user.password_hash, form.password.data):
            flash('Incorrect password', 'danger')
            return redirect(url_for('login'))
        login_user(user)
        next_page = request.args.get('next')
        return redirect(next_page or url_for('index'))
    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully', 'success')
    return redirect(url_for('index'))

def generate_churn_report(prediction):
    """Generate a comprehensive churn analysis report"""
    report = {
        'risk_factors': [],
        'retention_opportunities': [],
        'actionable_insights': []
    }
    if prediction.probability > 0.7:
        report['risk_factors'].append("High churn probability (>70%)")
    elif prediction.probability > 0.5:
        report['risk_factors'].append("Moderate churn probability (>50%)")
    if prediction.complaints_filed > 3:
        report['risk_factors'].append(f"High number of complaints ({prediction.complaints_filed})")
    if prediction.customer_support_calls > 5:
        report['risk_factors'].append(f"Frequent support calls ({prediction.customer_support_calls})")
    if prediction.tenure_months < 6:
        report['risk_factors'].append("Short customer tenure (less than 6 months)")
    if prediction.contract_type == 'Prepaid':
        report['retention_opportunities'].append("Offer postpaid conversion with benefits")
    if prediction.additional_services == 'None':
        report['retention_opportunities'].append("Recommend value-added services")
    if prediction.discount_offer_used == 'No':
        report['retention_opportunities'].append("Target with personalized discount offers")
    if prediction.billing_issues_reported > 0:
        report['actionable_insights'].append("Resolve billing issues immediately")
    if prediction.data_usage_gb < 1 and prediction.internet_service != 'None':
        report['actionable_insights'].append("Offer data usage guidance or packages")
    return report

# Routes: Predictions
def decode_to_int(value):
    """Safely convert a value to integer"""
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return 0
    return 0

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    form = PredictionForm()
    result = None
    prediction_id = None
    report = None
    if form.validate_on_submit():
        try:
            input_data = {
                'TelecomCompany': form.telecom_company.data,
                'Region': form.region.data,
                'Age': form.age.data,
                'Gender': form.gender.data,
                'ContractType': form.contract_type.data,
                'ContractDuration': form.contract_duration.data,
                'TenureMonths': form.tenure_months.data,
                'MonthlyCharges': form.monthly_charges.data,
                'DataUsageGB': form.data_usage_gb.data,
                'CallDurationMinutes': form.call_duration_minutes.data,
                'ComplaintsFiled': form.complaints_filed.data,
                'CustomerSupportCalls': form.customer_support_calls.data,
                'PaymentMethod': form.payment_method.data,
                'InternetService': form.internet_service.data,
                'AdditionalServices': form.additional_services.data,
                'DiscountOfferUsed': form.discount_offer_used.data,
                'BillingIssuesReported': form.billing_issues_reported.data
            }
            input_df = pd.DataFrame([input_data])
            processed_data = preprocess_data(input_df)
            prediction = model.predict(processed_data)[0]
            probability = model.predict_proba(processed_data)[0][1]
            new_prediction = Prediction(
                user_id=current_user.id,
                customer_id=None,
                telecom_company=form.telecom_company.data,
                region=form.region.data,
                age=form.age.data,
                gender=form.gender.data,
                contract_type=form.contract_type.data,
                contract_duration=form.contract_duration.data,
                tenure_months=form.tenure_months.data,
                monthly_charges=form.monthly_charges.data,
                data_usage_gb=form.data_usage_gb.data,
                call_duration_minutes=form.call_duration_minutes.data,
                complaints_filed=form.complaints_filed.data,
                customer_support_calls=form.customer_support_calls.data,
                payment_method=form.payment_method.data,
                internet_service=form.internet_service.data,
                additional_services=form.additional_services.data,
                discount_offer_used=form.discount_offer_used.data,
                billing_issues_reported=form.billing_issues_reported.data,
                prediction='Yes' if prediction == 1 else 'No',
                probability=probability,
                report_generated=True
            )
            db.session.add(new_prediction)
            db.session.commit()
            report = generate_churn_report(new_prediction)
            result = {
                'prediction': 'Yes' if prediction == 1 else 'No',
                'probability': probability,
                'recommendation': 'High risk of churn. Consider retention strategies.' if prediction == 1 else 'Low risk of churn.'
            }
            prediction_id = new_prediction.id
            if 'add_another' in request.form:
                new_form = PredictionForm(
                    telecom_company=form.telecom_company.data,
                    region=form.region.data,
                    gender=form.gender.data,
                    contract_type=form.contract_type.data,
                    contract_duration=form.contract_duration.data,
                    payment_method=form.payment_method.data,
                    internet_service=form.internet_service.data,
                    additional_services=form.additional_services.data,
                    discount_offer_used=form.discount_offer_used.data
                )
                return render_template('predict.html', form=new_form, result=result, prediction_id=prediction_id, report=report)
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Prediction error: {str(e)}")
            flash(f'Prediction failed: {str(e)}', 'danger')
    return render_template('predict.html', form=form, result=result, prediction_id=prediction_id, report=report)




@app.route('/api/csv-data')
def api_csv_data():
    try:
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'sample_data.csv'))
        return jsonify(df.to_dict('records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/show-csv')
def show_csv():
    try:
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'sample_data.csv'))
        return render_template('results.html', data=df.head(10).to_dict('records'))
    except Exception as e:
        return render_template('error.html', error=str(e)), 500

# New route for downloading CSV
@app.route('/load-csv')
def load_csv():
    try:
        csv_path = os.path.join(os.path.dirname(__file__), 'data', 'sample_data.csv')
        return send_file(csv_path, as_attachment=True, download_name='sample_data.csv')
    except Exception as e:
        return render_template('error.html', error=str(e)), 500

@app.route('/batch_predict', methods=['POST'])
@login_required
def batch_predict():
    form = PredictionForm()
    try:
        if 'file' not in request.files:
            flash('No file uploaded', 'danger')
            return render_template('predict.html', form=form, show_results=False)
        file = request.files['file']
        if not file or file.filename == '':
            flash('No file selected', 'danger')
            return render_template('predict.html', form=form, show_results=False)
        if not file.filename.endswith('.csv'):
            flash('Invalid file format. Please upload a CSV file.', 'danger')
            return render_template('predict.html', form=form, show_results=False)

        df = pd.read_csv(file, encoding='utf-8', dtype_backend='numpy_nullable')
        required_columns = [
            'TelecomCompany', 'Region', 'Age', 'Gender', 'ContractType',
            'ContractDuration', 'TenureMonths', 'MonthlyCharges', 'DataUsageGB',
            'CallDurationMinutes', 'ComplaintsFiled', 'CustomerSupportCalls',
            'PaymentMethod', 'InternetService', 'AdditionalServices',
            'DiscountOfferUsed', 'BillingIssuesReported'
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            flash(f'Missing required columns: {", ".join(missing_columns)}', 'danger')
            return render_template('predict.html', form=form, show_results=False)

        for col in required_columns:
            if df[col].isnull().any():
                if df[col].dtype in ['float64', 'int64', 'Int64', 'Float64']:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])

        numeric_columns = ['Age', 'TenureMonths', 'CallDurationMinutes', 'ComplaintsFiled',
                           'CustomerSupportCalls', 'BillingIssuesReported']
        for col in numeric_columns:
            df[col] = df[col].apply(decode_to_int)

        processed_data = preprocess_data(df)
        predictions = model.predict(processed_data)
        probabilities = model.predict_proba(processed_data)[:, 1]
        batch_id = str(uuid.uuid4())
        batch_results = []
        for idx, (pred, prob) in enumerate(zip(predictions, probabilities)):
            new_prediction = Prediction(
                user_id=current_user.id,
                batch_id=batch_id,
                customer_id=str(idx + 1),
                telecom_company=df.iloc[idx]['TelecomCompany'],
                region=df.iloc[idx]['Region'],
                age=decode_to_int(df.iloc[idx]['Age']),
                gender=df.iloc[idx]['Gender'],
                contract_type=df.iloc[idx]['ContractType'],
                contract_duration=df.iloc[idx]['ContractDuration'],
                tenure_months=decode_to_int(df.iloc[idx]['TenureMonths']),
                monthly_charges=float(df.iloc[idx]['MonthlyCharges']) if pd.notnull(df.iloc[idx]['MonthlyCharges']) else 0.0,
                data_usage_gb=float(df.iloc[idx]['DataUsageGB']) if pd.notnull(df.iloc[idx]['DataUsageGB']) else 0.0,
                call_duration_minutes=decode_to_int(df.iloc[idx]['CallDurationMinutes']),
                complaints_filed=decode_to_int(df.iloc[idx]['ComplaintsFiled']),
                customer_support_calls=decode_to_int(df.iloc[idx]['CustomerSupportCalls']),
                payment_method=df.iloc[idx]['PaymentMethod'],
                internet_service=df.iloc[idx]['InternetService'],
                additional_services=df.iloc[idx]['AdditionalServices'],
                discount_offer_used=df.iloc[idx]['DiscountOfferUsed'],
                billing_issues_reported=decode_to_int(df.iloc[idx]['BillingIssuesReported']),
                prediction='Yes' if pred == 1 else 'No',
                probability=prob,
                report_generated=True
            )
            db.session.add(new_prediction)
            batch_results.append({
                'customer_id': str(idx + 1),
                'telecom_company': df.iloc[idx]['TelecomCompany'],
                'region': df.iloc[idx]['Region'],
                'age': decode_to_int(df.iloc[idx]['Age']),
                'gender': df.iloc[idx]['Gender'],
                'contract_type': df.iloc[idx]['ContractType'],
                'contract_duration': df.iloc[idx]['ContractDuration'],
                'tenure_months': decode_to_int(df.iloc[idx]['TenureMonths']),
                'monthly_charges': float(df.iloc[idx]['MonthlyCharges']) if pd.notnull(df.iloc[idx]['MonthlyCharges']) else 0.0,
                'data_usage_gb': float(df.iloc[idx]['DataUsageGB']) if pd.notnull(df.iloc[idx]['DataUsageGB']) else 0.0,
                'call_duration_minutes': decode_to_int(df.iloc[idx]['CallDurationMinutes']),
                'complaints_filed': decode_to_int(df.iloc[idx]['ComplaintsFiled']),
                'customer_support_calls': decode_to_int(df.iloc[idx]['CustomerSupportCalls']),
                'payment_method': df.iloc[idx]['PaymentMethod'],
                'internet_service': df.iloc[idx]['InternetService'],
                'additional_services': df.iloc[idx]['AdditionalServices'],
                'discount_offer_used': df.iloc[idx]['DiscountOfferUsed'],
                'billing_issues_reported': decode_to_int(df.iloc[idx]['BillingIssuesReported']),
                'prediction': 'Yes' if pred == 1 else 'No',
                'probability': prob
            })
        db.session.commit()
        output_df = df.copy()
        output_df['Prediction'] = ['Yes' if pred == 1 else 'No' for pred in predictions]
        output_df['Probability'] = probabilities
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='w') as temp_file:
            output_df.to_csv(temp_file, index=False)
            temp_file_path = temp_file.name
        if not hasattr(app, 'batch_files'):
            app.batch_files = {}
        app.batch_files[batch_id] = temp_file_path
        flash('Batch prediction completed successfully', 'success')
        return render_template(
            'predict.html',
            form=form,
            batch_results=batch_results,
            batch_id=batch_id,
            show_results=True
        )
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Batch prediction error: {str(e)}")
        flash('Batch prediction failed. Please try again.', 'danger')
        return render_template('predict.html', form=form, show_results=False)

@app.route('/download_batch_results/<batch_id>')
@login_required
def download_batch_results(batch_id):
    try:
        if not hasattr(app, 'batch_files') or batch_id not in app.batch_files:
            flash('Batch results not found', 'danger')
            return redirect(url_for('predict'))
        temp_file_path = app.batch_files[batch_id]
        if not os.path.exists(temp_file_path):
            flash('Batch results file not found', 'danger')
            del app.batch_files[batch_id]
            return redirect(url_for('predict'))
        response = send_file(
            temp_file_path,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'batch_predictions_{batch_id}.csv'
        )
        os.remove(temp_file_path)
        del app.batch_files[batch_id]
        return response
    except Exception as e:
        app.logger.error(f"Download batch results error: {str(e)}")
        flash('Failed to download results. Please try again.', 'danger')
        return redirect(url_for('predict'))

@app.route('/download_sample_csv')
def download_sample_csv():
    try:
        columns = [
            'TelecomCompany', 'Region', 'Age', 'Gender', 'ContractType',
            'ContractDuration', 'TenureMonths', 'MonthlyCharges', 'DataUsageGB',
            'CallDurationMinutes', 'ComplaintsFiled', 'CustomerSupportCalls',
            'PaymentMethod', 'InternetService', 'AdditionalServices',
            'DiscountOfferUsed', 'BillingIssuesReported'
        ]
        sample_data = [
            {
                'TelecomCompany': 'Airtel', 'Region': 'Dar es Salaam', 'Age': 35, 'Gender': 'Male',
                'ContractType': 'Postpaid', 'ContractDuration': '12 Months', 'TenureMonths': 24,
                'MonthlyCharges': 65.50, 'DataUsageGB': 10.5, 'CallDurationMinutes': 120,
                'ComplaintsFiled': 1, 'CustomerSupportCalls': 2, 'PaymentMethod': 'Credit Card',
                'InternetService': 'Fiber', 'AdditionalServices': 'Streaming', 'DiscountOfferUsed': 'Yes',
                'BillingIssuesReported': 0
            },
            {
                'TelecomCompany': 'Vodacom', 'Region': 'Mwanza', 'Age': 28, 'Gender': 'Female',
                'ContractType': 'Prepaid', 'ContractDuration': '1 Month', 'TenureMonths': 12,
                'MonthlyCharges': 45.00, 'DataUsageGB': 5.0, 'CallDurationMinutes': 80,
                'ComplaintsFiled': 0, 'CustomerSupportCalls': 1, 'PaymentMethod': 'Mobile Money',
                'InternetService': 'Mobile Data', 'AdditionalServices': 'None', 'DiscountOfferUsed': 'No',
                'BillingIssuesReported': 1
            }
        ]
        df = pd.DataFrame(sample_data, columns=columns)
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name='sample_batch_prediction.csv'
        )
    except Exception as e:
        app.logger.error(f"Sample CSV download error: {str(e)}")
        flash('Failed to download sample CSV. Please try again.', 'danger')
        return redirect(url_for('predict'))

@app.route('/predictions')
@login_required
def predictions():
    user_predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.desc()).all()
    return render_template('predictions.html', predictions=user_predictions)

@app.route('/delete_prediction/<int:prediction_id>', methods=['POST'])
@login_required
def delete_prediction(prediction_id):
    prediction = Prediction.query.get_or_404(prediction_id)
    if prediction.user_id != current_user.id:
        flash('Unauthorized action', 'danger')
        return redirect(url_for('predictions'))
    try:
        db.session.delete(prediction)
        db.session.commit()
        flash('Prediction deleted successfully', 'success')
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Delete prediction error: {str(e)}")
        flash('Failed to delete prediction. Please try again.', 'danger')
    return redirect(url_for('predictions'))

@app.route('/update_prediction/<int:prediction_id>', methods=['GET', 'POST'])
@login_required
def update_prediction(prediction_id):
    prediction = Prediction.query.get_or_404(prediction_id)
    if prediction.user_id != current_user.id:
        flash('Unauthorized action', 'danger')
        return redirect(url_for('predictions'))
    form = PredictionForm(obj=prediction)
    if form.validate_on_submit():
        try:
            form.populate_obj(prediction)
            db.session.commit()
            flash('Prediction updated successfully', 'success')
            return redirect(url_for('predictions'))
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Update prediction error: {str(e)}")
            flash('Failed to update prediction. Please try again.', 'danger')
    return render_template('update_prediction.html', form=form, prediction=prediction)

@app.route('/analysis')
@login_required
def analysis():
    try:
        viz_data = create_visualizations()
        return render_template('analysis.html', viz_data=viz_data)
    except Exception as e:
        app.logger.error(f"Analysis error: {str(e)}")
        flash('Failed to load analysis. Please try again.', 'danger')
        return redirect(url_for('index'))

@app.route('/export_predictions/<format>')
@login_required
def export_predictions(format):
    try:
        predictions = Prediction.query.filter_by(user_id=current_user.id).all()
        if not predictions:
            flash('No predictions found to export', 'warning')
            return redirect(url_for('predictions'))

        def safe_float(value):
            try:
                return float(value) if value is not None else 0.0
            except (ValueError, TypeError):
                return 0.0

        def safe_int(value):
            try:
                return int(value) if value is not None else 0
            except (ValueError, TypeError):
                return 0

        def safe_str(value):
            return str(value) if value is not None else 'N/A'

        data = [{
            'ID': pred.id,
            'Date': safe_str(pred.timestamp.strftime('%Y-%m-%d %H:%M:%S') if pred.timestamp else 'N/A'),
            'Telecom Company': safe_str(pred.telecom_company),
            'Region': safe_str(pred.region),
            'Age': safe_int(pred.age),
            'Gender': safe_str(pred.gender),
            'Contract Type': safe_str(pred.contract_type),
            'Contract Duration': safe_str(pred.contract_duration),
            'Tenure (Months)': safe_int(pred.tenure_months),
            'Monthly Charges': safe_float(pred.monthly_charges),
            'Data Usage (GB)': safe_float(pred.data_usage_gb),
            'Call Duration (Minutes)': safe_int(pred.call_duration_minutes),
            'Complaints Filed': safe_int(pred.complaints_filed),
            'Customer Support Calls': safe_int(pred.customer_support_calls),
            'Payment Method': safe_str(pred.payment_method),
            'Internet Service': safe_str(pred.internet_service),
            'Additional Services': safe_str(pred.additional_services),
            'Discount Offer Used': safe_str(pred.discount_offer_used),
            'Billing Issues Reported': safe_int(pred.billing_issues_reported),
            'Prediction': safe_str(pred.prediction),
            'Probability (%)': round(safe_float(pred.probability) * 100, 2)
        } for pred in predictions]

        df = pd.DataFrame(data)

        if format == 'csv':
            output = io.BytesIO()
            df.to_csv(output, index=False, encoding='utf-8')
            output.seek(0)
            return send_file(
                output,
                mimetype='text/csv',
                as_attachment=True,
                download_name='churn_predictions.csv'
            )
        elif format == 'excel':
            try:
                import openpyxl
            except ImportError:
                app.logger.error("openpyxl not installed")
                flash('Excel export failed: openpyxl is not installed.', 'danger')
                return redirect(url_for('predictions'))
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Predictions')
            output.seek(0)
            return send_file(
                output,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                as_attachment=True,
                download_name='churn_predictions.xlsx'
            )
        else:
            flash('Invalid export format', 'danger')
            return redirect(url_for('predictions'))
    except Exception as e:
        app.logger.error(f"Export error: {str(e)}", exc_info=True)
        flash(f'Export failed: {str(e)}', 'danger')
        return redirect(url_for('predictions'))

def generate_pdf_report(prediction, report):
    """Generate a detailed PDF report with human-readable values"""
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)

    field_mappings = {
        'telecom_company': {
            'Airtel': 'Airtel', 'Tigo': 'Tigo', 'Vodacom': 'Vodacom',
            'Halotel': 'Halotel', 'TTCL': 'TTCL', 'Zantel': 'Zantel'
        },
        'region': {
            'Arusha': 'Arusha', 'Dar es Salaam': 'Dar es Salaam', 'Dodoma': 'Dodoma',
            'Geita': 'Geita', 'Iringa': 'Iringa', 'Kagera': 'Kagera',
            'Katavi': 'Katavi', 'Kigoma': 'Kigoma', 'Kilimanjaro': 'Kilimanjaro',
            'Lindi': 'Lindi', 'Manyara': 'Manyara', 'Mara': 'Mara',
            'Mbeya': 'Mbeya', 'Morogoro': 'Morogoro', 'Mtwara': 'Mtwara',
            'Mwanza': 'Mwanza', 'Njombe': 'Njombe', 'Pemba North': 'Pemba North',
            'Pemba South': 'Pemba South', 'Pwani': 'Pwani', 'Rukwa': 'Rukwa',
            'Ruvuma': 'Ruvuma', 'Shinyanga': 'Shinyanga', 'Simiyu': 'Simiyu',
            'Singida': 'Singida', 'Songwe': 'Songwe', 'Tabora': 'Tabora',
            'Tanga': 'Tanga', 'Unguja North': 'Unguja North', 'Unguja South': 'Unguja South'
        },
        'gender': {'Male': 'Male', 'Female': 'Female'},
        'contract_type': {'Prepaid': 'Prepaid', 'Postpaid': 'Postpaid', 'Hybrid': 'Hybrid'},
        'contract_duration': {
            '1 Month': '1 Month', '3 Months': '3 Months', '6 Months': '6 Months',
            '12 Months': '12 Months', '24 Months': '24 Months'
        },
        'payment_method': {
            'Credit Card': 'Credit Card', 'Bank Transfer': 'Bank Transfer',
            'Mobile Money': 'Mobile Money', 'Cash': 'Cash', 'Voucher': 'Voucher', 'Other': 'Other'
        },
        'internet_service': {
            'Mobile Data': 'Mobile Data', 'Fiber': 'Fiber', 'DSL': 'DSL',
            'WiMAX': 'WiMAX', 'None': 'None'
        },
        'additional_services': {
            'Streaming': 'Streaming', 'VPN': 'VPN', 'Cloud Storage': 'Cloud Storage',
            'Gaming': 'Gaming', 'None': 'None'
        },
        'discount_offer_used': {'Yes': 'Yes', 'No': 'No'},
        'prediction': {'Yes': 'Will Churn', 'No': 'Will Not Churn'}
    }

    def safe_value(value):
        """Return a safe string representation of a value"""
        return str(value) if value is not None else 'N/A'

    def get_display_value(field_name, value):
        """Return human-readable value for a field"""
        if field_name in field_mappings:
            return field_mappings[field_name].get(safe_value(value), safe_value(value))
        return safe_value(value)

    def format_currency(value):
        """Format a number as currency"""
        try:
            return f"TZS {float(value):,.2f}"
        except (TypeError, ValueError):
            return 'N/A'

    def format_float(value, decimals=2):
        """Format a float with specified decimal places"""
        try:
            return f"{float(value):.{decimals}f}"
        except (TypeError, ValueError):
            return 'N/A'

    # Header
    p.setFont("Helvetica-Bold", 16)
    p.drawString(100, 750, "Tanzania Telecom Churn Prediction Report")
    p.setFont("Helvetica", 12)
    p.drawString(100, 730, f"Report Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
    p.drawString(100, 710, f"Prediction ID: {prediction.id}")

    # Customer Details
    p.setFont("Helvetica-Bold", 14)
    p.drawString(100, 680, "Customer Details:")
    p.setFont("Helvetica", 12)
    details = [
        f"Customer ID: {safe_value(prediction.customer_id)}",
        f"Telecom Company: {get_display_value('telecom_company', prediction.telecom_company)}",
        f"Region: {get_display_value('region', prediction.region)}",
        f"Age: {safe_value(prediction.age)}",
        f"Gender: {get_display_value('gender', prediction.gender)}",
        f"Contract Type: {get_display_value('contract_type', prediction.contract_type)}",
        f"Contract Duration: {get_display_value('contract_duration', prediction.contract_duration)}",
        f"Tenure (Months): {safe_value(prediction.tenure_months)}",
        f"Monthly Charges: {format_currency(prediction.monthly_charges)}",
        f"Data Usage (GB): {format_float(prediction.data_usage_gb)}",
        f"Call Duration (Minutes): {safe_value(prediction.call_duration_minutes)}",
        f"Complaints Filed: {safe_value(prediction.complaints_filed)}",
        f"Customer Support Calls: {safe_value(prediction.customer_support_calls)}",
        f"Payment Method: {get_display_value('payment_method', prediction.payment_method)}",
        f"Internet Service: {get_display_value('internet_service', prediction.internet_service)}",
        f"Additional Services: {get_display_value('additional_services', prediction.additional_services)}",
        f"Discount Offer Used: {get_display_value('discount_offer_used', prediction.discount_offer_used)}",
        f"Billing Issues Reported: {safe_value(prediction.billing_issues_reported)}"
    ]
    y_position = 660
    for detail in details:
        p.drawString(120, y_position, detail)
        y_position -= 20

    # Prediction Results
    p.setFont("Helvetica-Bold", 14)
    p.drawString(100, y_position - 20, "Prediction Results:")
    p.setFont("Helvetica", 12)
    p.drawString(120, y_position - 40, f"Churn Prediction: {get_display_value('prediction', prediction.prediction)}")
    p.drawString(120, y_position - 60, f"Probability: {format_float(prediction.probability * 100, 2)}%")

    # Risk Analysis
    p.setFont("Helvetica-Bold", 14)
    p.drawString(100, y_position - 100, "Risk Analysis:")
    p.setFont("Helvetica", 12)
    if report['risk_factors']:
        p.drawString(120, y_position - 120, "Key Risk Factors:")
        y_position -= 140
        for factor in report['risk_factors']:
            p.drawString(140, y_position, f"- {factor}")
            y_position -= 20
    else:
        p.drawString(120, y_position - 120, "No significant risk factors identified")
        y_position -= 140

    # Recommendations
    recommendations = [
        "Immediate retention actions recommended:" if prediction.prediction == 'Yes' else "Maintenance actions recommended:",
        "- Offer personalized discounts or promotions" if prediction.prediction == 'Yes' else "- Continue current service quality",
        "- Assign dedicated account manager" if prediction.prediction == 'Yes' else "- Monitor for early warning signs",
        "- Resolve any outstanding complaints" if prediction.prediction == 'Yes' else "- Offer loyalty rewards",
        "- Provide value-added services trial" if prediction.prediction == 'Yes' else "- Proactive customer check-ins"
    ]
    p.setFont("Helvetica-Bold", 14)
    p.drawString(100, y_position - 20, "Recommendations:")
    p.setFont("Helvetica", 12)
    for rec in recommendations:
        p.drawString(120, y_position - 40, rec)
        y_position -= 20

    # Retention Opportunities
    if report['retention_opportunities']:
        p.setFont("Helvetica-Bold", 14)
        p.drawString(100, y_position - 40, "Retention Opportunities:")
        p.setFont("Helvetica", 12)
        y_position -= 60
        for opp in report['retention_opportunities']:
            p.drawString(120, y_position, f"- {opp}")
            y_position -= 20

    # Actionable Insights
    if report['actionable_insights']:
        p.setFont("Helvetica-Bold", 14)
        p.drawString(100, y_position - 20, "Actionable Insights:")
        p.setFont("Helvetica", 12)
        y_position -= 40
        for insight in report['actionable_insights']:
            p.drawString(120, y_position, f"- {insight}")
            y_position -= 20

    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer

@app.route('/download_pdf/<int:prediction_id>')
@login_required
def download_pdf(prediction_id):
    prediction = Prediction.query.get_or_404(prediction_id)
    if prediction.user_id != current_user.id:
        flash('Unauthorized access', 'danger')
        return redirect(url_for('predictions'))
    try:
        report = generate_churn_report(prediction)
        pdf_buffer = generate_pdf_report(prediction, report)
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'churn_report_{prediction.id}.pdf'
        )
    except Exception as e:
        app.logger.error(f"PDF generation error: {str(e)}")
        flash('Failed to generate PDF report. Please try again.', 'danger')
        return redirect(url_for('predictions'))

# API Routes
@csrf.exempt
@app.route('/api/predictions', methods=['GET'])
@login_required
def api_predictions():
    try:
        page = request.args.get('page', 1, type=int)
        per_page = 50
        predictions = Prediction.query.filter_by(user_id=current_user.id)\
            .order_by(Prediction.timestamp.desc())\
            .paginate(page=page, per_page=per_page, error_out=False)
        results = [{
            'id': p.id,
            'telecom_company': p.telecom_company,
            'region': p.region,
            'age': p.age,
            'tenure_months': p.tenure_months,
            'prediction': p.prediction,
            'probability': float(p.probability),
            'timestamp': p.timestamp.isoformat()
        } for p in predictions.items]
        return jsonify({
            'results': results,
            'status': 'success',
            'total_pages': predictions.pages,
            'current_page': page
        })
    except Exception as e:
        app.logger.error(f"API predictions error: {str(e)}")
        return jsonify({'error': 'Failed to fetch predictions', 'status': 'error'}), 500

@csrf.exempt
@app.route('/api/predictions/<int:prediction_id>', methods=['DELETE'])
@login_required
def api_delete_prediction(prediction_id):
    try:
        prediction = Prediction.query.get_or_404(prediction_id)
        if prediction.user_id != current_user.id:
            return jsonify({'error': 'Unauthorized action', 'status': 'error'}), 403
        db.session.delete(prediction)
        db.session.commit()
        return jsonify({'status': 'success'})
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"API delete prediction error: {str(e)}")
        return jsonify({'error': 'Failed to delete prediction', 'status': 'error'}), 500

@csrf.exempt
@app.route('/api/predictions', methods=['DELETE'])
@login_required
def delete_all_predictions():
    try:
        Prediction.query.filter_by(user_id=current_user.id).delete()
        db.session.commit()
        return jsonify({'status': 'success'})
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"API delete all predictions error: {str(e)}")
        return jsonify({'error': 'Failed to delete predictions', 'status': 'error'}), 500

@csrf.exempt
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        required_fields = [
            'TelecomCompany', 'Region', 'Age', 'Gender', 'ContractType',
            'ContractDuration', 'TenureMonths', 'MonthlyCharges', 'DataUsageGB',
            'CallDurationMinutes', 'ComplaintsFiled', 'CustomerSupportCalls',
            'PaymentMethod', 'InternetService', 'AdditionalServices',
            'DiscountOfferUsed', 'BillingIssuesReported'
        ]
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        input_df = pd.DataFrame([data])
        processed_data = preprocess_data(input_df)
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0][1]
        return jsonify({
            'prediction': 'Yes' if prediction == 1 else 'No',
            'probability': float(probability),
            'status': 'success'
        })
    except Exception as e:
        app.logger.error(f"API predict error: {str(e)}")
        return jsonify({'error': 'Prediction failed', 'status': 'error'}), 500

@csrf.exempt
@app.route('/api/batch_predict', methods=['POST'])
def api_batch_predict():
    try:
        data = request.get_json()
        if not isinstance(data, list):
            return jsonify({'error': 'Input must be a list of records'}), 400
        required_fields = [
            'CustomerID', 'TelecomCompany', 'Region', 'Age', 'Gender', 'ContractType',
            'ContractDuration', 'TenureMonths', 'MonthlyCharges', 'DataUsageGB',
            'CallDurationMinutes', 'ComplaintsFiled', 'CustomerSupportCalls',
            'PaymentMethod', 'InternetService', 'AdditionalServices',
            'DiscountOfferUsed', 'BillingIssuesReported'
        ]
        for record in data:
            missing_fields = [field for field in required_fields if field not in record]
            if missing_fields:
                return jsonify({'error': f'Missing required fields: {", ".join(missing_fields)}'}), 400
        input_df = pd.DataFrame(data)
        processed_data = preprocess_data(input_df)
        predictions = model.predict(processed_data)
        probabilities = model.predict_proba(processed_data)[:, 1]
        results = [{
            'customer_id': str(input_df.iloc[idx]['CustomerID']),
            'prediction': 'Yes' if pred == 1 else 'No',
            'probability': float(prob)
        } for idx, (pred, prob) in enumerate(zip(predictions, probabilities))]
        return jsonify({
            'results': results,
            'status': 'success'
        })
    except Exception as e:
        app.logger.error(f"API batch predict error: {str(e)}")
        return jsonify({'error': 'Batch prediction failed', 'status': 'error'}), 500

if __name__ == '__main__':
    app.run(debug=True)