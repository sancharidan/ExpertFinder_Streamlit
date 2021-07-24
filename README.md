# ExpertFinder

Steps to deply on Google Cloud Run

- Command to build the application
```
gcloud builds submit --tag gcr.io/expertfinderstreamlit/ExpertFinderStreamlit  --project=expertfinderstreamlit
```
- Command to deploy the application
```
gcloud run deploy --image gcr.io/expertfinderstreamlit/ExpertFinderStreamlit --platform managed  --project=expertfinderstreamlit --allow-unauthenticated
```
_________________________________________________________

Steps to run the app locally:
```
# clone repository
git clone https://github.com/sancharidan/ExpertFinder.git
cd ExpertFinder

# create python environment
conda create --name expertfinder python=3.8.8

# activate conda environment
conda activate expertfinder

# install requirements 
pip install -r requirements.txt

# run the Streamlit app
streamlit run app.py
```

Navigate to ```http://localhost:8501``` to find the Streamlit App as below

<img src="https://github.com/sancharidan/ExpertFinder/blob/a806407f371128185b1027cdec77ee553cd23efd/ExpertFinder%20Streamlit.PNG" width="600">
