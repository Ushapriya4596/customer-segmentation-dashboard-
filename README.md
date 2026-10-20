# Customer Segmentation using K-Means

A beginner-friendly Python + Machine Learning project that groups customers with similar income, spending, age, and purchase frequency.

## Project files
- `data/customers.csv` — included demo dataset (320 fictional customers)
- `customer_segmentation.py` — full script for analysis, clustering, graphs, and insights
- `requirements.txt` — required Python packages
- `outputs/` — generated charts and CSV results after you run the script

## Features
1. Loads and checks customer data
2. Scales features so measurements on different ranges are comparable
3. Uses the Elbow Method to inspect a reasonable cluster count
4. Trains K-Means clustering
5. Saves customer cluster labels and a segment summary
6. Generates visualizations
7. Prints practical marketing suggestions for each cluster

## Setup (Windows)
1. Install Python 3.10+ from https://www.python.org/downloads/ (select "Add Python to PATH" during installation).
2. Extract this folder to a location you can find.
3. Open Command Prompt inside the `customer_segmentation_project` folder.
4. Run:
   ```bash
   py -m pip install -r requirements.txt
   py customer_segmentation.py
   ```
   If `py` doesn't work, try `python` instead.

## Run in Google Colab
1. Open https://colab.research.google.com/
2. Upload `customer_segmentation.py` and `data/customers.csv`, or upload the project ZIP and extract it.
3. Run the script from a notebook cell with the right paths, or paste the code into cells and set `DATA_PATH` to the uploaded CSV location.

## Important
The dataset is synthetic and intended for learning. It does not represent real customers. The script uses 5 clusters as a starter choice; compare the elbow chart and segment summaries, and try different values of `K` to understand the effect.

## How to explain the project
- Problem: A business has many customers but wants to understand groups with different shopping patterns.
- Method: K-Means, an unsupervised learning algorithm, groups customers using scaled numeric features.
- Features: Age, annual income, spending score, annual purchase frequency.
- Output: A cluster label per customer, segment summaries, and graphs.
- Use: Businesses can tailor loyalty offers and campaigns to each segment. The model does not prove why customers behave a certain way; it only groups similar records.

## Launch the interactive dashboard

1. Open Command Prompt in this project folder.
2. Install the dashboard libraries: `py -m pip install -r requirements.txt`
3. Make sure the results exist. If needed, run: `py customer_segmentation.py`
4. Start the dashboard: `py -m streamlit run streamlit_app.py`
5. A browser tab should open. If not, copy the `Local URL` shown in Command Prompt into your browser.

Keep `streamlit_app.py` in the main project folder, next to the `data` and `outputs` folders.
