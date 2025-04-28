from flask import Flask, render_template, request, redirect, url_for, session, send_file
import os
import pandas as pd
import time
from datetime import datetime
import numpy as np
import smtplib
from email.message import EmailMessage
import requests

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
            amount_threshold = session.get('rule_amount', 0)
            result_df, metrics = module.detect_fraud(df.copy(), amount_threshold)
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

def get_user_info():
    ip = request.remote_addr or 'Unknown IP'
    username = session.get('username', 'Guest')

    try:
        res = requests.get(f"https://ipinfo.io/{ip}/json")
        if res.status_code == 200:
            data = res.json()
            city = data.get('city', '')
            country = data.get('country', '')
            location = f"{city}, {country}" if city and country else "Unknown Location"
        else:
            location = "Unknown Location"
    except Exception:
        location = "Unknown Location"

    return {'ip': ip, 'username': username, 'location': location}

@app.route('/')
def home():
    user_info = get_user_info()
    return render_template('index.html', title="GOAT - Home", user_info=user_info)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'admin':
            session['username'] = username  # <-- 🔥 Save username in session
            return redirect(url_for('upload'))
        else:
            return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

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

@app.route('/processing')
def processing():
    filepath = session.get('uploaded_file')
    if not filepath:
        return redirect(url_for('upload'))
    run_all_algorithms(filepath)
    return redirect(url_for('algorithm'))

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
def loading(algo):
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
    user_info = get_user_info()
    return "<h4>PDF download coming soon!</h4>"

@app.route('/contact')
def contact():
    user_info = get_user_info()
    return "<h2 style='text-align:center;margin-top:2rem;'>Contact us at goat@security.ai</h2>"

@app.route('/send_email', methods=['POST'])
def send_email():
    name = request.form.get('name')
    email = request.form.get('email')

    if not name or not email:
        return "Name and email required!", 400

    msg = EmailMessage()
    msg['Subject'] = "Your Fraud Detection Report"
    msg['From'] = 'goatreportdev@gmail.com'
    msg['To'] = email

    message_body = f"""
    Greetings {name},

    Your fraud detection report is ready!

    Thank you for using GOAT Fraud Detection Service.

    (Note: PDF report feature is coming soon!)

    Best Regards,
    GOAT Team 🐐
    """

    msg.set_content(message_body)

    try:
        server = smtplib.SMTP('smtp-relay.brevo.com', 587)
        server.starttls()
        server.login('8b8abb001@smtp-brevo.com', '1EN9m06WsGqL8Rht')
        server.send_message(msg)
        server.quit()
        return "<h3>Email sent successfully! 📩</h3><a href='/'>Go Home</a>"
    except Exception as e:
        return f"Failed to send email: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
