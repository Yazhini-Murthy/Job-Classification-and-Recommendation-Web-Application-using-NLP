from flask import Flask, render_template, redirect,request
from gensim.models import FastText
import pandas as pd
import pickle
import os
import json
from bs4 import BeautifulSoup
import numpy as np

job_data = []
sales_jobs = []
engineering_jobs = []
accounting_jobs = []
healthcare_jobs = []

with open("job_data.json", "r") as f:
    job_data = json.load(f)

def docvecs(embeddings, docs):
    vecs = np.zeros((len(docs), embeddings.vector_size))
    for i, doc in enumerate(docs):
        valid_keys = [term for term in doc if term in embeddings.key_to_index]
        docvec = np.vstack([embeddings[term] for term in valid_keys])
        docvec = np.sum(docvec, axis=0)
        vecs[i,:] = docvec
    return vecs

    
def categorize_jobs(job_data):
    for job in job_data:
        category = job["job_category"]
        if category == "Sales":
            sales_jobs.append(job)
        elif category == "Engineering":
            engineering_jobs.append(job)
        elif category == "Accounting_Finance":
            accounting_jobs.append(job)
        elif category == "Healthcare_Nursing":
            healthcare_jobs.append(job)

    return {
        "Sales": sales_jobs,
        "Engineering": engineering_jobs,
        "Accounting": accounting_jobs,
        "Healthcare": healthcare_jobs
    }

def save_job_data():
    with open("job_data.json", "w") as f:
        json.dump(job_data, f, indent=4)

categorized_jobs = categorize_jobs(job_data)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('job_list.html', jobs = job_data)

@app.route('/job/<int:job_index>')
def display_job_details(job_index):
    # Assuming you have a list of job_data
    job = None
    for j in job_data:
        if j['custom_index'] == job_index:
            job = j
            break

    if job:
        return render_template('job_details.html', job=job)
    else:
        # Handle the case where the job is not found (e.g., show a 404 page or a message)
        return "Job not found"
    
@app.route('/sales')
def display_sales_jobs():
    return render_template('job_list.html', category="Sales", jobs=categorized_jobs["Sales"])

@app.route('/engineering')
def display_engineering_jobs():
    return render_template('job_list.html', category="Engineering", jobs=categorized_jobs["Engineering"])

@app.route('/accounting')
def display_accounting_jobs():
    return render_template('job_list.html', category="Accounting", jobs=categorized_jobs["Accounting"])

@app.route('/healthcare')
def display_healthcare_jobs():
    return render_template('job_list.html', category="Healthcare", jobs=categorized_jobs["Healthcare"])

@app.route('/admin')
def admin():
    return redirect('/classify.html')

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    global job_data
    if request.method == 'POST':

            # Read the content
            f_title = request.form['title']
            f_content = request.form['description']
            f_salary = request.form['salary']

            # Classify the content
            if request.form['button'] == 'Classify':

                tokenized_data = f_content.split(' ')

                # Load the FastText model
                jobFT = FastText.load("desc_FT.model")
                jobFT_wv= jobFT.wv
               # Generate vector representation of the tokenized data
                jobFT_dvs = docvecs(jobFT_wv, [tokenized_data])

                # Load the LR model
                pkl_filename = "descFT_LR.pkl"
                with open(pkl_filename, 'rb') as file:
                    model = pickle.load(file)

                # Predict the label of tokenized_data
                y_pred = model.predict(jobFT_dvs)
                y_pred = y_pred[0]
                return render_template('classify.html', prediction=y_pred, title=f_title, description=f_content, salary = f_salary)

            elif request.form['button'] == 'Save':
                # Check if the recommended category is empty
                cat_recommend = request.form['category']
                if cat_recommend == '':
                    return render_template('classify.html', prediction=cat_recommend,
                                            title=f_title, description=f_content, salary=f_salary,
                                            category_flag='Recommended category must not be empty.')

                elif cat_recommend not in ['Accounting_Finance', 'Engineering', 'Sales', 'Healthcare_Nursing']:
                    return render_template('classify.html', prediction=cat_recommend,
                                            title=f_title, description=f_content, salary=f_salary,
                                            category_flag='Recommended category must belong to: Accounting_Finance, Sales, Engineering, Healthcare Nursing.')

                else:
                    # Create a new job dictionary
                    new_job = {
                        "job_category": cat_recommend,
                        "job_title": f_title, # Update with the company name
                        "job_desc": f_content,
                        "job_salary":f_salary,
                        "custom_index": len(job_data)  # Assign a unique index based on the length of job_data
                    }
                    job_data.append(new_job)
                    save_job_data()

                    if new_job['job_category'] == "Sales":
                        sales_jobs.append(new_job)
                    elif new_job['job_category']  == "Engineering":
                        engineering_jobs.append(new_job)
                    elif new_job['job_category']  == "Accounting_Finance":
                        accounting_jobs.append(new_job)
                    elif new_job['job_category']  == "Healthcare_Nursing":
                        healthcare_jobs.append(new_job)

                    # Redirect to the newly-generated news article or any other page as needed
                    return redirect('/job/'+str(new_job['custom_index']))   
    else:
        # Handle the GET request, which may involve rendering an initial form
        return render_template('classify.html')

@app.route('/<folder>/<filename>')
def article(folder, filename):
    return render_template(folder + '/' + filename)

@app.route('/search', methods = ['POST'])
def search():

  if request.method == 'POST':
    if request.form['search'] == 'Search':
        search_string = request.form["searchword"]
        
        # Search for the string within job descriptions in the job_data
        job_search_results = []
        for job in job_data:
            if search_string in job['job_desc'] or search_string in job['job_title']:
                job_search_results.append(job)

        num_results = len(job_search_results)

        return render_template('search.html', num_results=num_results, search_string=search_string, job_search=job_search_results)

    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)


