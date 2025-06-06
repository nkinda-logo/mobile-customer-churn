from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, validators, SelectField, FloatField, IntegerField
from datetime import datetime
import os
import pandas as pd
import numpy as np
import joblib
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from utils.data_processing import preprocess_data
from utils.visualization import create_visualizations
import config

app = Flask(__name__)
app.config.from_object(config.Config)

# Database setup
db_dir = os.path.join(os.path.dirname(__file__), 'database')
os.makedirs(db_dir, exist_ok=True)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///' + os.path.join(db_dir, 'app.db'))
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    is_admin = db.Column(db.Boolean, default=False)
    predictions = db.relationship('Prediction', backref='user', lazy=True)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    customer_id = db.Column(db.Integer)
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


# Forms
class RegistrationForm(FlaskForm):
    username = StringField('Username', [validators.Length(min=4, max=25)])
    email = StringField('Email Address', [validators.Length(min=6, max=100)])
    password = PasswordField('Password', [
        validators.DataRequired(),
        validators.EqualTo('confirm', message='Passwords must match')
    ])
    confirm = PasswordField('Repeat Password')

class LoginForm(FlaskForm):
    username = StringField('Username', [validators.DataRequired()])
    password = PasswordField('Password', [validators.DataRequired()])

class PredictionForm(FlaskForm):
    telecom_company = SelectField('Telecom Company', choices=[
        ('Airtel', 'Airtel'), 
        ('Tigo', 'Tigo'), 
        ('Vodacom', 'Vodacom'),
        ('Halotel', 'Halotel'),
        ('TTCL', 'TTCL'),
        ('Zantel', 'Zantel'),
    ], validators=[validators.DataRequired()])
    
    region = SelectField('Region', choices=[
        ('Arusha', 'Arusha'),
        ('Dar es Salaam', 'Dar es Salaam'),
        ('Dodoma', 'Dodoma'),
        ('Geita', 'Geita'),
        ('Iringa', 'Iringa'),
        ('Kagera', 'Kagera'),
        ('Katavi', 'Katavi'),
        ('Kigoma', 'Kigoma'),
        ('Kilimanjaro', 'Kilimanjaro'),
        ('Lindi', 'Lindi'),
        ('Manyara', 'Manyara'),
        ('Mara', 'Mara'),
        ('Mbeya', 'Mbeya'),
        ('Morogoro', 'Morogoro'),
        ('Mtwara', 'Mtwara'),
        ('Mwanza', 'Mwanza'),
        ('Njombe', 'Njombe'),
        ('Pemba North', 'Pemba North'),
        ('Pemba South', 'Pemba South'),
        ('Pwani', 'Pwani'),
        ('Rukwa', 'Rukwa'),
        ('Ruvuma', 'Ruvuma'),
        ('Shinyanga', 'Shinyanga'),
        ('Simiyu', 'Simiyu'),
        ('Singida', 'Singida'),
        ('Songwe', 'Songwe'),
        ('Tabora', 'Tabora'),
        ('Tanga', 'Tanga'),
        ('Unguja North', 'Unguja North'),
        ('Unguja South', 'Unguja South')
    ], validators=[validators.DataRequired()])
    
    age = IntegerField('Age', [validators.NumberRange(min=18, max=110)])
    gender = SelectField('Gender', choices=[('Male', 'Male'), ('Female', 'Female')], validators=[validators.DataRequired()])
    
    contract_type = SelectField('Contract Type', choices=[
        ('Prepaid', 'Prepaid'), 
        ('Postpaid', 'Postpaid'),
        ('Hybrid', 'Hybrid')
    ], validators=[validators.DataRequired()])
    
    contract_duration = SelectField('Contract Duration', choices=[
        ('1 Month', '1 Month'),
        ('3 Months', '3 Months'),
        ('6 Months', '6 Months'),
        ('12 Months', '12 Months'),
        ('24 Months', '24 Months')
    ], validators=[validators.DataRequired()])
    
    tenure_months = IntegerField('Tenure (Months)', [validators.NumberRange(min=1, max=120)])
    monthly_charges = FloatField('Monthly Charges', [validators.NumberRange(min=0)])
    data_usage_gb = FloatField('Data Usage (GB)', [validators.NumberRange(min=0)])
    call_duration_minutes = IntegerField('Call Duration (Minutes)', [validators.NumberRange(min=0)])
    complaints_filed = IntegerField('Complaints Filed', [validators.NumberRange(min=0)])
    customer_support_calls = IntegerField('Customer Support Calls', [validators.NumberRange(min=0)])
    
    payment_method = SelectField('Payment Method', choices=[
        ('Credit Card', 'Credit Card'),
        ('Bank Transfer', 'Bank Transfer'),
        ('Mobile Money', 'Mobile Money'),
        ('Cash', 'Cash'),
        ('Voucher', 'Voucher'),
        ('Other', 'Other')
    ], validators=[validators.DataRequired()])
    
    internet_service = SelectField('Internet Service', choices=[
        ('Mobile Data', 'Mobile Data'),
        ('Fiber', 'Fiber'),
        ('DSL', 'DSL'),
        ('WiMAX', 'WiMAX'),
        ('None', 'None'),
    ], validators=[validators.DataRequired()])
    
    additional_services = SelectField('Additional Services', choices=[
        ('Streaming', 'Streaming'),
        ('VPN', 'VPN'),
        ('Cloud Storage', 'Cloud Storage'),
        ('Gaming', 'Gaming'),
        ('None', 'None'),
    ], validators=[validators.DataRequired()])
    
    discount_offer_used = SelectField('Discount Offer Used', choices=[
        ('Yes', 'Yes'),
        ('No', 'No')
    ], validators=[validators.DataRequired()])
    
    billing_issues_reported = IntegerField('Billing Issues Reported', [validators.NumberRange(min=0)])

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def load_model():
    """Load the ML model with robust error handling and fallback"""
    try:
        model = joblib.load('models/churn_model.pkl')
        app.logger.info("Loaded primary churn prediction model")
        return model
    except Exception as e:
        app.logger.error(f"Failed to load primary model: {str(e)}")
        try:
            model = joblib.load('models/backup_churn_model.pkl')
            app.logger.warning("Using backup churn prediction model")
            return model
        except Exception as e:
            app.logger.critical(f"Failed to load backup model: {str(e)}")
            raise RuntimeError("No working model available")

model = load_model()

def initialize_database():
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(username='admin').first():
            admin = User(
                username='admin',
                email='admin@example.com',
                password_hash=generate_password_hash('admin123'),
                is_admin=True
            )
            db.session.add(admin)
            db.session.commit()

initialize_database()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

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
                form.password.data,
                method='pbkdf2:sha256',
                salt_length=16
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
            app.logger.error(f"Registration error: {e}")
            flash(f'Registration failed: {e}', 'danger')
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
        return redirect(next_page) if next_page else redirect(url_for('index'))
    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
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
                'probability': f"{probability * 100:.2f}%",
                'recommendation': 'High risk of churn. Consider retention strategies.' if prediction == 1 else 'Low risk of churn.',
                'report': report
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

@app.route('/batch_predict', methods=['POST'])
@login_required
def batch_predict():
    try:
        # Check if a file was uploaded
        if 'file' not in request.files:
            flash('No file uploaded', 'danger')
            return redirect(url_for('predict'))

        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'danger')
            return redirect(url_for('predict'))

        if file and file.filename.endswith('.csv'):
            # Read and validate CSV file
            try:
                df = pd.read_csv(file)
            except Exception as e:
                flash(f'Error reading CSV file: {str(e)}', 'danger')
                return redirect(url_for('predict'))

            required_columns = [
                'TelecomCompany', 'Region', 'Age', 'Gender', 'ContractType',
                'ContractDuration', 'TenureMonths', 'MonthlyCharges', 'DataUsageGB',
                'CallDurationMinutes', 'ComplaintsFiled', 'CustomerSupportCalls',
                'PaymentMethod', 'InternetService', 'AdditionalServices',
                'DiscountOfferUsed', 'BillingIssuesReported'
            ]

            # Validate columns
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                flash(f'Missing required columns: {", ".join(missing_columns)}', 'danger')
                return redirect(url_for('predict'))

            # Preprocess data
            processed_data = preprocess_data(df)

            # Make batch predictions
            predictions = model.predict(processed_data)
            probabilities = model.predict_proba(processed_data)[:, 1]

            # Save predictions to database
            prediction_ids = []
            for idx, (pred, prob) in enumerate(zip(predictions, probabilities)):
                new_prediction = Prediction(
                    user_id=current_user.id,
                    telecom_company=df.iloc[idx]['TelecomCompany'],
                    region=df.iloc[idx]['Region'],
                    age=df.iloc[idx]['Age'],
                    gender=df.iloc[idx]['Gender'],
                    contract_type=df.iloc[idx]['ContractType'],
                    contract_duration=df.iloc[idx]['ContractDuration'],
                    tenure_months=df.iloc[idx]['TenureMonths'],
                    monthly_charges=df.iloc[idx]['MonthlyCharges'],
                    data_usage_gb=df.iloc[idx]['DataUsageGB'],
                    call_duration_minutes=df.iloc[idx]['CallDurationMinutes'],
                    complaints_filed=df.iloc[idx]['ComplaintsFiled'],
                    customer_support_calls=df.iloc[idx]['CustomerSupportCalls'],
                    payment_method=df.iloc[idx]['PaymentMethod'],
                    internet_service=df.iloc[idx]['InternetService'],
                    additional_services=df.iloc[idx]['AdditionalServices'],
                    discount_offer_used=df.iloc[idx]['DiscountOfferUsed'],
                    billing_issues_reported=df.iloc[idx]['BillingIssuesReported'],
                    prediction='Yes' if pred == 1 else 'No',
                    probability=prob,
                    report_generated=True
                )
                db.session.add(new_prediction)
                prediction_ids.append(new_prediction.id)

            db.session.commit()

            # Create output DataFrame with predictions
            output_df = df.copy()
            output_df['Prediction'] = ['Yes' if pred == 1 else 'No' for pred in predictions]
            output_df['Probability'] = probabilities
            output_df['PredictionID'] = prediction_ids

            # Generate CSV output
            output = BytesIO()
            output_df.to_csv(output, index=False)
            output.seek(0)

            flash('Batch prediction completed successfully', 'success')
            return send_file(
                output,
                mimetype='text/csv',
                as_attachment=True,
                download_name='batch_predictions.csv'
            )

        else:
            flash('Invalid file format. Please upload a CSV file.', 'danger')
            return redirect(url_for('predict'))

    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Batch prediction error: {str(e)}")
        flash(f'Batch prediction failed: {str(e)}', 'danger')
        return redirect(url_for('predict'))

@app.route('/predictions')
@login_required
def predictions():
    user_predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.desc()).all()
    return render_template('predictions.html', predictions=user_predictions)

@app.route('/analysis')
@login_required
def analysis():
    viz_data = create_visualizations()
    return render_template('analysis.html', viz_data=viz_data)
  

@app.route('/export_predictions/<format>')
@login_required
def export_predictions(format):
    try:
        predictions = Prediction.query.filter_by(user_id=current_user.id).all()
        
        if not predictions:
            flash('No predictions found to export', 'warning')
            return redirect(url_for('predictions'))
        
        data = []
        for pred in predictions:
            data.append({
                'Timestamp': pred.timestamp,
                'Telecom Company': pred.telecom_company,
                'Region': pred.region,
                'Age': pred.age,
                'Gender': pred.gender,
                'Contract Type': pred.contract_type,
                'Contract Duration': pred.contract_duration,
                'Tenure (Months)': pred.tenure_months,
                'Monthly Charges': pred.monthly_charges,
                'Data Usage (GB)': pred.data_usage_gb,
                'Call Duration (Minutes)': pred.call_duration_minutes,
                'Complaints Filed': pred.complaints_filed,
                'Customer Support Calls': pred.customer_support_calls,
                'Payment Method': pred.payment_method,
                'Internet Service': pred.internet_service,
                'Additional Services': pred.additional_services,
                'Discount Offer Used': pred.discount_offer_used,
                'Billing Issues Reported': pred.billing_issues_reported,
                'Prediction': pred.prediction,
                'Probability': pred.probability
            })
        
        df = pd.DataFrame(data)
        
        if format == 'csv':
            output = BytesIO()
            df.to_csv(output, index=False)
            output.seek(0)
            return send_file(
                output,
                mimetype='text/csv',
                as_attachment=True,
                download_name='churn_predictions.csv'
            )
        elif format == 'excel':
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
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
        flash('Export failed. Please try again.', 'danger')
        app.logger.error(f"Export error: {str(e)}")
        return redirect(url_for('predictions'))

def generate_pdf_report(prediction, report):
    """Generate a detailed PDF report"""
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    
    p.setFont("Helvetica-Bold", 16)
    p.drawString(100, 750, "Tanzania Telecom Churn Prediction Report")
    p.setFont("Helvetica", 12)
    p.drawString(100, 730, f"Report Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
    p.drawString(100, 710, f"Prediction ID: {prediction.id}")
    
    p.setFont("Helvetica-Bold", 14)
    p.drawString(100, 680, "Customer Details:")
    p.setFont("Helvetica", 12)
    
    details = [
        f"Telecom Company: {prediction.telecom_company}",
        f"Region: {prediction.region}",
        f"Age: {prediction.age}",
        f"Gender: {prediction.gender}",
        f"Contract Type: {prediction.contract_type}",
        f"Contract Duration: {prediction.contract_duration}",
        f"Tenure (Months): {prediction.tenure_months}",
        f"Monthly Charges: {prediction.monthly_charges}",
        f"Data Usage (GB): {prediction.data_usage_gb}",
        f"Call Duration (Minutes): {prediction.call_duration_minutes}",
        f"Complaints Filed: {prediction.complaints_filed}",
        f"Customer Support Calls: {prediction.customer_support_calls}",
        f"Payment Method: {prediction.payment_method}",
        f"Internet Service: {prediction.internet_service}",
        f"Additional Services: {prediction.additional_services}",
        f"Discount Offer Used: {prediction.discount_offer_used}",
        f"Billing Issues Reported: {prediction.billing_issues_reported}"
    ]
    
    y_position = 660
    for detail in details:
        p.drawString(120, y_position, detail)
        y_position -= 20
    
    p.setFont("Helvetica-Bold", 14)
    p.drawString(100, y_position - 20, "Prediction Results:")
    p.setFont("Helvetica", 12)
    p.drawString(120, y_position - 40, f"Churn Prediction: {prediction.prediction}")
    p.drawString(120, y_position - 60, f"Probability: {prediction.probability * 100:.2f}%")
    
    if report['risk_factors']:
        p.setFont("Helvetica-Bold", 14)
        p.drawString(100, y_position - 100, "Risk Analysis:")
        p.setFont("Helvetica", 12)
        p.drawString(120, y_position - 120, "Key Risk Factors:")
        y_position -= 140
        for factor in report['risk_factors']:
            p.drawString(140, y_position, f"- {factor}")
            y_position -= 20
    else:
        p.drawString(120, y_position - 120, "No significant risk factors identified")
        y_position -= 140
    
    if prediction.prediction == 'Yes':
        recommendations = [
            "Immediate retention actions recommended:",
            "- Offer personalized discounts or promotions",
            "- Assign dedicated account manager",
            "- Resolve any outstanding complaints",
            "- Provide value-added services trial",
            "- Conduct satisfaction survey"
        ]
    else:
        recommendations = [
            "Maintenance actions recommended:",
            "- Continue current service quality",
            "- Monitor for early warning signs",
            "- Offer loyalty rewards",
            "- Proactive customer check-ins"
        ]
    
    p.setFont("Helvetica-Bold", 14)
    p.drawString(100, y_position - 20, "Recommendations:")
    p.setFont("Helvetica", 12)
    
    for rec in recommendations:
        p.drawString(120, y_position - 40, rec)
        y_position -= 20
    
    if report['retention_opportunities']:
        p.setFont("Helvetica-Bold", 14)
        p.drawString(100, y_position - 40, "Retention Opportunities:")
        p.setFont("Helvetica", 12)
        y_position -= 60
        for opp in report['retention_opportunities']:
            p.drawString(120, y_position, f"- {opp}")
            y_position -= 20
    
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
    
    report = generate_churn_report(prediction)
    pdf_buffer = generate_pdf_report(prediction, report)
    
    return send_file(
        pdf_buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f'churn_report_{prediction.id}.pdf'
    )

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
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/batch_predict', methods=['POST'])
def api_batch_predict():
    try:
        data = request.get_json()
        
        if not isinstance(data, list):
            return jsonify({'error': 'Input must be a list of records'}), 400

        required_fields = [
            'TelecomCompany', 'Region', 'Age', 'Gender', 'ContractType',
            'ContractDuration', 'TenureMonths', 'MonthlyCharges', 'DataUsageGB',
            'CallDurationMinutes', 'ComplaintsFiled', 'CustomerSupportCalls',
            'PaymentMethod', 'InternetService', 'AdditionalServices',
            'DiscountOfferUsed', 'BillingIssuesReported'
        ]

        # Validate each record
        for record in data:
            missing_fields = [field for field in required_fields if field not in record]
            if missing_fields:
                return jsonify({'error': f'Missing required fields in record: {", ".join(missing_fields)}'}), 400

        input_df = pd.DataFrame(data)
        processed_data = preprocess_data(input_df)

        predictions = model.predict(processed_data)
        probabilities = model.predict_proba(processed_data)[:, 1]

        results = []
        for idx, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                'record_index': idx,
                'prediction': 'Yes' if pred == 1 else 'No',
                'probability': float(prob)
            })

        return jsonify({
            'results': results,
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(username='admin').first():
            admin = User(
                username='admin',
                email='admin@example.com',
                password_hash=generate_password_hash('admin123'),
                is_admin=True
            )
            db.session.add(admin)
            db.session.commit()
    
    app.run(debug=True)
    