from flask import Flask, render_template, request, redirect, url_for, session, send_file
import os
import pandas as pd
import time
from datetime import datetime
import numpy as np
# from utils.helper import generate_pdf  # Uncomment later when email/pdf is implemented

app = Flask(__name__)
app.secret_key = 'supersecretkey'

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALGORITHMS = [
    'z_score',
    'isolation_forest',
    'lof',
    'autoencoder',
    'kmeans'
]

# 🔥 Global dictionary to store outputs (not session)
outputs_store = {}

def run_all_algorithms(filepath):
    import pandas as pd
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

        # 🔥 Save output in global variable
        outputs_store[algo] = result_df.to_dict(orient='records')

        print(f"{algo} completed successfully.")

    session['summaries'] = summaries  # ✅ only small summaries in session
    session['labeled_data'] = outputs_store.get('rule_based', [])  # Save something for analysis

@app.route('/')
def home():
    return render_template('index.html', title="GOAT - Home")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'admin':
            return redirect(url_for('upload'))
        else:
            return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            session['uploaded_file'] = filepath
            return redirect(url_for('processing'))
        else:
            return render_template('upload.html', error='Please upload a valid CSV file.')
    return render_template('upload.html')

@app.route('/processing')
def processing():
    filepath = session.get('uploaded_file')
    if not filepath:
        return redirect(url_for('upload'))

    run_all_algorithms(filepath)
    return redirect(url_for('algorithm'))  # ✅ After processing, go to algorithm placards

@app.route('/algorithm')
def algorithm():
    summaries = session.get('summaries', {})
    return render_template('algorithm.html', summaries=summaries)

@app.route('/rule_define', methods=['GET', 'POST'])
def rule_define():
    if request.method == 'POST':
        fields = request.form.getlist('field')
        operators = request.form.getlist('operator')
        values = request.form.getlist('value')

        rules = []
        for f, op, val in zip(fields, operators, values):
            rules.append({'field': f, 'operator': op, 'value': val})

        session['rule_conditions'] = rules
        return redirect(url_for('results', algo='rule_based'))

    return render_template('rule_define.html')

@app.route('/loading/<algo>')
def loading(algo):
    return render_template('loading.html', algo=algo)

@app.route('/results/<algo>')
def results(algo):
    table = outputs_store.get(algo, [])
    session['current_algo'] = algo

    # 🔥 FIX: update labeled_data dynamically
    session['labeled_data'] = table

    return render_template('results.html', table=table, algo=algo)

@app.route('/analysis')
def analysis():
    import plotly.graph_objs as go
    from plotly.offline import plot
    import pandas as pd

    data = session.get('labeled_data', [])
    if not data:
        return redirect(url_for('upload'))

    df = pd.DataFrame(data)

    if 'Fraud' not in df.columns:
        df['Fraud'] = 'Non-Fraud'  # 🔥 Add dummy column if missing

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

    return render_template('analysis.html', plot_div=plot_div, details=details)


@app.route('/awareness')
def awareness():
    return render_template('awareness.html')

@app.route('/download')
def download():
    return "<h4>PDF download coming soon!</h4>"

@app.route('/contact')
def contact():
    return "<h2 style='text-align:center;margin-top:2rem;'>Contact us at goat@security.ai</h2>"

if __name__ == '__main__':
    app.run(debug=True)
