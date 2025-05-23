import os
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'  # 👈 ALLOW HTTP FOR LOCAL TESTING



from flask import Flask, render_template, request, redirect, url_for, session, send_file
import os
import pandas as pd
import time
from datetime import datetime
import numpy as np
import smtplib
from email.message import EmailMessage
import requests
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle, SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.fonts import addMapping
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from requests_oauthlib import OAuth2Session

#MongoDB Database connection
from pymongo import MongoClient

# Connect to local MongoDB
client = MongoClient("mongodb://localhost:27017/")

# Access the database and collection
db = client["FraudDB"]
users_collection = db["users"]           # for login/signup
transactions_collection = db["transactions"]  # for fraud data

GOOGLE_CLIENT_ID = "42611484438-iv1d8lv8rhdt2pubbn398344jmjf98ln.apps.googleusercontent.com"
GOOGLE_CLIENT_SECRET = "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"
REDIRECT_URI = "http://localhost:5000/callback"

AUTHORIZATION_BASE_URL = 'https://accounts.google.com/o/oauth2/auth'
TOKEN_URL = 'https://accounts.google.com/o/oauth2/token'
USER_INFO_URL = 'https://www.googleapis.com/oauth2/v1/userinfo'

SCOPE = ['https://www.googleapis.com/auth/userinfo.profile', 'https://www.googleapis.com/auth/userinfo.email']
USER_INFO = USER_INFO_URL  # Fixing name mismatch


app = Flask(__name__)
app.secret_key = 'supersecretkey'

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALGORITHMS = ['z_score', 'isolation_forest', 'lof', 'autoencoder', 'kmeans']
outputs_store = {}

def run_all_algorithms(filepath):
    df = pd.read_csv(filepath)
    algorithms = ['z_score', 'isolation_forest', 'lof', 'kmeans', 'autoencoder', 'rule_based']
    summaries = {}

    for algo in algorithms:
        print(f"Running {algo}...")
        start = time.time()
        module = __import__(f"algorithms.{algo}", fromlist=[None])

        if algo == 'rule_based':
         rules = session.get('rules', [])
         result_df, metrics = module.detect_fraud(df.copy(), rules)
        else:
            result_df, metrics = module.detect_fraud(df.copy())

        end = time.time()

        clean_details = {}
        if metrics.get('details'):
            for k, v in metrics['details'].items():
                if isinstance(v, (np.integer, np.int64, np.int32)):
                    clean_details[k] = int(v)
                elif isinstance(v, (np.floating, np.float64, np.float32)):
                    clean_details[k] = float(v)
                else:
                    clean_details[k] = v

        summaries[algo] = {
            'anomalies': float(metrics.get('anomaly_percent', 0)),
            'accuracy': float(metrics.get('accuracy', 0)),
            'precision': float(metrics.get('precision', 0)),
            'time': round(end - start, 2),
            'details': clean_details
        }

        outputs_store[algo] = result_df.to_dict(orient='records')
        print(f"{algo} completed successfully.")

    session['summaries'] = summaries
    session['labeled_data'] = outputs_store.get('rule_based', [])
    
# changed location
def get_user_info():
    ip = request.remote_addr or 'Unknown IP'
    username = session.get('username', 'Guest')

    try:
        res = requests.get(f"https://ipinfo.io/{ip}/json")
        if res.status_code == 200:
            data = res.json()
            city = data.get('city', '')
            country = data.get('country', '')
            location = f"{city}, {country}" if city and country else "Dehradun"
        else:
            location = "Unknown Location"
    except Exception:
        location = "Unknown Location"

    return {'ip': ip, 'username': username, 'location': location}

#signup page
from flask import request, redirect
from pymongo import MongoClient
import bcrypt

client = MongoClient('mongodb://localhost:27017/')
db = client['fraud_detection']
users_collection = db['users']

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']

        if users_collection.find_one({'username': username}):
            return render_template('signup.html', message='Username already exists.')

        hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        users_collection.insert_one({
            'name': name,
            'email': email,
            'username': username,
            'password': hashed_pw
        })
        return render_template('signup.html', message='You have successfully registered!')

    return render_template('signup.html')



@app.route('/')
def home():
    user_info = get_user_info()
    return render_template('index.html', title="GOAT - Home", user_info=user_info)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = users_collection.find_one({'username': username})
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
            session['user'] = user['username']
            return redirect('/upload')
        else:
            return render_template('login.html', error='Invalid credentials.')

    return render_template('login.html')





@app.route('/google_login')
def google_login():
    google = OAuth2Session(GOOGLE_CLIENT_ID, scope=SCOPE, redirect_uri=REDIRECT_URI)
    authorization_url, state = google.authorization_url(
        AUTHORIZATION_BASE_URL,
        access_type="offline", prompt="select_account")

    session['oauth_state'] = state
    return redirect(authorization_url)



@app.route('/callback')
def callback():
    from requests_oauthlib import OAuth2Session

    if 'oauth_state' not in session:
        return redirect(url_for('login'))

    google = OAuth2Session(GOOGLE_CLIENT_ID, redirect_uri=REDIRECT_URI, state=session['oauth_state'])

    try:
        token = google.fetch_token(
            TOKEN_URL,
            client_secret=GOOGLE_CLIENT_SECRET,
            authorization_response=request.url
        )
    except Exception as e:
        return f"<h3>Failed to fetch token:</h3><p>{e}</p>"

    try:
        resp = google.get(USER_INFO_URL)
        user_info = resp.json()
        # Save user info to session
        session['username'] = user_info.get('name') or user_info.get('email') or 'Google User'
    except Exception as e:
        session['username'] = 'Google User'
        print(f"Failed to fetch user info: {e}")

    return redirect(url_for('upload'))


#upload route
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    user_info = get_user_info()
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            session['uploaded_file'] = filepath
            return redirect(url_for('processing'))
        else:
            return render_template('upload.html', error='Please upload a valid CSV file.', user_info=user_info)
    return render_template('upload.html', user_info=user_info)




@app.route('/quiz')
def quiz():
    user_info = get_user_info()
    return render_template('quiz.html', user_info=user_info)

@app.route('/loading')
def loading():
    return render_template('loading.html')

#processing
@app.route('/processing')
def processing():
    filepath = session.get('uploaded_file')
    if not filepath:
        return redirect(url_for('upload'))

    return render_template('processing.html')  # shows loading animation

    # NOTE: JavaScript in processing.html will redirect to /run-algos

    
@app.route('/run-algos')
def run_algos():
    filepath = session.get('uploaded_file')
    if not filepath:
        return redirect(url_for('upload'))
    
    run_all_algorithms(filepath)
    return redirect(url_for('algorithm'))  # Redirect to placards


@app.route('/algorithm')
def algorithm():
    user_info = get_user_info()
    summaries = session.get('summaries', {})
    return render_template('algorithm.html', summaries=summaries, user_info=user_info)

@app.route('/rule_define', methods=['GET', 'POST'])
def rule_define():
    user_info = get_user_info()
    if request.method == 'POST':
        fields = request.form.getlist('field')
        operators = request.form.getlist('operator')
        values = request.form.getlist('value')

        rules = []
        for f, op, val in zip(fields, operators, values):
            rules.append({'field': f, 'operator': op, 'value': val})

        session['rule_conditions'] = rules
        return redirect(url_for('results', algo='rule_based'))

    return render_template('rule_define.html', user_info=user_info)

@app.route('/loading/<algo>')
def loading_algorithm(algo):
    user_info = get_user_info()
    return render_template('loading.html', algo=algo, user_info=user_info)


@app.route('/results/<algo>')
def results(algo):
    user_info = get_user_info()
    table = outputs_store.get(algo, [])
    session['current_algo'] = algo
    session['labeled_data'] = table
    return render_template('results.html', table=table, algo=algo, user_info=user_info)

@app.route('/analysis')
def analysis():
    user_info = get_user_info()
    import plotly.graph_objs as go
    from plotly.offline import plot
    import pandas as pd

    data = session.get('labeled_data', [])
    if not data:
        return redirect(url_for('upload'))

    df = pd.DataFrame(data)

    if 'Fraud' not in df.columns:
        df['Fraud'] = 'Non-Fraud'

    counts = df['Fraud'].value_counts()

    if counts.empty:
        labels = ['Fraud', 'Non-Fraud']
        values = [1, 1]
    else:
        labels = counts.index
        values = counts.values

    trace = go.Pie(labels=labels, values=values, hole=0.4)

    layout = go.Layout(
        title='Fraud vs Non-Fraud Transactions',
        height=500,
        width=700
    )

    fig = go.Figure(data=[trace], layout=layout)
    plot_div = plot(fig, output_type='div')

    details = None
    if session.get('summaries') and session.get('current_algo'):
        details = session['summaries'][session['current_algo']].get('details', None)

    return render_template('analysis.html', plot_div=plot_div, details=details, user_info=user_info)

@app.route('/awareness')
def awareness():
    user_info = get_user_info()
    return render_template('awareness.html', user_info=user_info)



@app.route('/download')
def download():
    from fpdf import FPDF
    import pandas as pd

    table = session.get('labeled_data', [])
    algo = session.get('current_algo', 'algorithm')

    if not table:
        return "<h4>No data to download!</h4>"

    class PDF(FPDF):
        def header(self):
            # GOAT Logo
            self.image('static/images/logo.png', 10, 8, 15)  # x, y, width
            self.set_font('Times', 'B', 24)
            self.set_text_color(43, 58, 103)
            self.cell(0, 20, "GOAT - Fraud Detection Report", ln=True, align='C')
            self.ln(10)

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font('Times', '', 12)
    pdf.set_text_color(0, 0, 0)

    # Table header
    headers = list(table[0].keys())
    col_widths = [pdf.get_string_width(header) + 10 for header in headers]

    for idx, header in enumerate(headers):
        if header.lower() == 'time':
            col_widths[idx] = 45

    pdf.set_fill_color(200, 220, 255)
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 10, header, border=1, align='C', fill=True)
    pdf.ln()

    # Table rows
    for row in table:
        for i, (key, value) in enumerate(row.items()):
            if key == 'Fraud':
                if value == 'Fraud':
                    pdf.set_fill_color(255, 200, 200)
                else:
                    pdf.set_fill_color(200, 255, 200)
            else:
                pdf.set_fill_color(255, 255, 255)

            pdf.cell(col_widths[i], 10, str(value), border=1, align='C', fill=True)
        pdf.ln()

    filename = f"GOAT_Report_{algo}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    filepath = os.path.join('uploads', filename)
    pdf.output(filepath)

    return send_file(filepath, as_attachment=True)

#contact page route
@app.route('/contact')
def contact():
    user_info = get_user_info()
    return render_template('contactpage.html', user_info=user_info)

from fpdf import FPDF
import tempfile

@app.route('/send_email', methods=['POST'])
def send_email():
    name = request.form.get('name')
    email = request.form.get('email')

    if not name or not email:
        return "Name and email required!", 400

    # Step 1: Generate PDF report
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"GOAT Fraud Detection Report", ln=True, align='C')
    pdf.ln(10)
    pdf.multi_cell(0, 10, f"Hello {name},\n\nBelow is a brief summary of your fraud detection results:\n\n")

    # Load summary data from session
    summaries = session.get('summaries', {})
    for algo, details in summaries.items():
        pdf.cell(0, 10, f"Algorithm: {algo}", ln=True)
        pdf.cell(0, 10, f" - Anomaly %: {details.get('anomalies', 0)}%", ln=True)
        pdf.cell(0, 10, f" - Accuracy: {details.get('accuracy', 0)}%", ln=True)
        pdf.cell(0, 10, f" - Precision: {details.get('precision', 0)}%", ln=True)
        pdf.cell(0, 10, f" - Time Taken: {details.get('time', 0)} sec", ln=True)
        pdf.ln(5)

    pdf.multi_cell(0, 10, "\nThank you for using GOAT Fraud Detection Service.\n Stay safe, stay smart!")

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        pdf_path = tmp_file.name
        pdf.output(pdf_path)

    # Step 2: Compose Email
    msg = EmailMessage()
    msg['Subject'] = "Your Fraud Detection Report"
    msg['From'] = 'goatreportdev@gmail.com'
    msg['To'] = email

    msg.set_content(f"""
    Greetings {name},

    Your fraud detection report is attached as a PDF.

    Thank you for using GOAT Fraud Detection Service.

    Best Regards,
    GOAT Team 
    """)

    # Attach the PDF file
    with open(pdf_path, 'rb') as f:
        pdf_data = f.read()
        msg.add_attachment(pdf_data, maintype='application', subtype='pdf', filename='GOAT_Fraud_Report.pdf')

    # Step 3: Send using Brevo SMTP
    try:
        server = smtplib.SMTP('smtp-relay.brevo.com', 587)
        server.starttls()
        server.login('8b8abb001@smtp-brevo.com', '1EN9m06WsGqL8Rht')
        server.send_message(msg)
        server.quit()
        return "<h3>Email sent successfully with your PDF report! 📄📩</h3><a href='/'>Go Home</a>"
    except Exception as e:
        return f"Failed to send email: {str(e)}"




@app.route('/visuals')
def visuals():
    import plotly.graph_objs as go
    from plotly.offline import plot
    import pandas as pd

    data = session.get('labeled_data', [])
    if not data:
        return redirect(url_for('upload'))

    df = pd.DataFrame(data)
    plot_divs = {}

    # 1. Pie chart - Fraud vs Non-Fraud
    pie_counts = df['Fraud'].value_counts()
    pie_colors = ['#d60000' if label == 'Fraud' else '#007a3d' for label in pie_counts.index]
    pie_trace = go.Pie(labels=pie_counts.index, values=pie_counts.values, hole=0.4, marker=dict(colors=pie_colors))
    pie_fig = go.Figure(data=[pie_trace])
    pie_fig.update_layout(title='Fraud vs Non-Fraud (Pie)')
    plot_divs["Fraud vs Non-Fraud (Pie)"] = plot(pie_fig, output_type='div')

    # 2. Bar chart - Count by Location with proper color coding
    if 'location' in df.columns:
        bar_data = df.groupby(['location', 'Fraud']).size().unstack(fill_value=0)
        bar_colors = {'Fraud': '#d60000', 'Non-Fraud': '#007a3d'}
        bar_trace = [
            go.Bar(
                name=label,
                x=bar_data.index,
                y=bar_data[label],
                marker=dict(color=bar_colors.get(label, 'gray'))
            ) for label in bar_data.columns
        ]
        bar_fig = go.Figure(data=bar_trace)
        bar_fig.update_layout(
            barmode='group',
            title='Transaction Count by Location',
            xaxis_title="Location",
            yaxis_title="Count"
        )
        plot_divs["Transaction Count by Location"] = plot(bar_fig, output_type='div')

    # 3. Line chart - Amount over Time
    if 'time' in df.columns and 'amount' in df.columns:
        try:
            df['parsed_time'] = pd.to_datetime(df['time'], errors='coerce')
            line_df = df.dropna(subset=['parsed_time'])
            line_df = line_df.sort_values(by='parsed_time')
            line_trace = go.Scatter(
                x=line_df['parsed_time'],
                y=line_df['amount'],
                mode='lines',
                line=dict(color='#2b3a67')
            )
            line_fig = go.Figure(data=[line_trace])
            line_fig.update_layout(
                title='Transaction Amount Over Time',
                xaxis_title="Time",
                yaxis_title="Transaction Amount"
            )
            plot_divs["Transaction Amount Over Time"] = plot(line_fig, output_type='div')
        except Exception as e:
            print(f"Line chart error: {e}")

    # 4. Heatmap - Correlation Matrix
    if df.select_dtypes(include='number').shape[1] > 1:
        corr = df.select_dtypes(include='number').corr()
        heatmap_trace = go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='Viridis'
        )
        heatmap_fig = go.Figure(data=[heatmap_trace])
        heatmap_fig.update_layout(title='Feature Correlation Heatmap')
        plot_divs["Feature Correlation Heatmap"] = plot(heatmap_fig, output_type='div')

    return render_template('visuals.html', plot_divs=plot_divs)


@app.route('/rule_based', methods=['GET', 'POST'])
def rule_based():
    if request.method == 'POST':
        df = session.get('data_df')
        if not df:
            return redirect(url_for('upload'))

        # Extract rules from form
        fields = request.form.getlist('field')
        operators = request.form.getlist('operator')
        values = request.form.getlist('value')

        rules = []
        for f, o, v in zip(fields, operators, values):
            rules.append({'field': f, 'operator': o, 'value': v})

        df = pd.DataFrame(df)
        from rule_based import detect_fraud
        result_df, stats = detect_fraud(df.copy(), rules)

        session['results_table'] = result_df.to_dict(orient='records')
        session['summary'] = stats
        session['algo'] = 'rule_based'

        return redirect(url_for('results'))

    return render_template('rule_define.html')





if __name__ == '__main__':
    app.run(debug=True)
