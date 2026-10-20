import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title='Customer Segmentation Dashboard', page_icon='📊', layout='wide')
BASE = Path(__file__).resolve().parent

st.markdown('''<style>
.main {background:#f6f8fc}
[data-testid="stMetric"] {background:white;border:1px solid #e5eaf5;padding:16px;border-radius:14px}
h1,h2,h3 {color:#172b4d}
</style>''', unsafe_allow_html=True)

@st.cache_data
def load_data():
    path = BASE / 'outputs' / 'customer_segments.csv'
    if not path.exists():
        path = BASE / 'data' / 'customers.csv'
    df = pd.read_csv(path)
    rename = {}
    for col in df.columns:
        key = col.lower().replace('_','').replace(' ','')
        if key in ('customerid','customer','id'): rename[col] = 'CustomerID'
        elif key == 'age': rename[col] = 'Age'
        elif key in ('annualincome','annualincomekusd','income'): rename[col] = 'AnnualIncome'
        elif key in ('spendingscore','spending'): rename[col] = 'SpendingScore'
        elif key in ('purchases','purchasecount'): rename[col] = 'Purchases'
        elif key in ('cluster','segment'): rename[col] = 'Cluster'
    return df.rename(columns=rename)

df = load_data()
if 'Cluster' not in df.columns:
    st.error('Cluster column is missing. First run: py customer_segmentation.py')
    st.stop()
for col in ['CustomerID','Age','AnnualIncome','SpendingScore']:
    if col not in df.columns:
        st.error(f'Missing expected column: {col}. Check your dataset.')
        st.stop()
if 'Purchases' not in df.columns: df['Purchases'] = np.nan

df['Cluster'] = df['Cluster'].astype(str)
clusters = sorted(df['Cluster'].unique(), key=lambda x: int(x) if x.isdigit() else x)
st.sidebar.title('📊 Customer Insights')
st.sidebar.caption('Python • Machine Learning • K-Means')
page = st.sidebar.radio('Navigate', ['Overview','Customer Segments','Detailed Data','Insights & Recommendations'])
selected = st.sidebar.multiselect('Filter customer groups', clusters, default=clusters)
d = df[df['Cluster'].isin(selected)].copy()
st.title('Customer Segmentation Dashboard')
st.write('Understand customer groups. Discover patterns. Make smarter decisions.')

if page == 'Overview':
    a,b,c,e = st.columns(4)
    a.metric('Total customers', f'{len(d):,}')
    b.metric('Customer segments', f'{d.Cluster.nunique():,}')
    c.metric('Average annual income', f'{d.AnnualIncome.mean():,.1f}')
    e.metric('Average spending score', f'{d.SpendingScore.mean():.1f}')
    left,right = st.columns([1.2,1])
    with left:
        st.subheader('Customer groups')
        fig = px.scatter(d, x='AnnualIncome', y='SpendingScore', color='Cluster', hover_data=['CustomerID','Age','Purchases'], template='plotly_white', labels={'AnnualIncome':'Annual income (dataset units)','SpendingScore':'Spending score'})
        fig.update_layout(height=420, legend_title='Cluster')
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader('Customers in each segment')
        counts = d.Cluster.value_counts().sort_index().rename_axis('Cluster').reset_index(name='Customers')
        fig = px.bar(counts, x='Cluster', y='Customers', color='Cluster', text='Customers', template='plotly_white')
        fig.update_layout(height=420, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    st.subheader('Segment profile')
    cols = [x for x in ['Age','AnnualIncome','SpendingScore','Purchases'] if x in d.columns]
    st.dataframe(d.groupby('Cluster')[cols].mean().round(2), use_container_width=True)
elif page == 'Customer Segments':
    st.subheader('Explore a customer segment')
    if d.empty: st.info('Choose at least one group in the sidebar.')
    else:
        chosen = st.selectbox('Select cluster', sorted(d.Cluster.unique()))
        group = d[d.Cluster == chosen]
        a,b,c = st.columns(3)
        a.metric('Customers in group', len(group))
        b.metric('Average age', f'{group.Age.mean():.1f}')
        c.metric('Average spending score', f'{group.SpendingScore.mean():.1f}')
        st.write('Customer records')
        st.dataframe(group, use_container_width=True, hide_index=True)
elif page == 'Detailed Data':
    st.subheader('Search and inspect customer records')
    query = st.text_input('Search by customer ID or any value')
    view = d.copy()
    if query:
        mask = view.astype(str).apply(lambda col: col.str.contains(query, case=False, na=False)).any(axis=1)
        view = view[mask]
    st.caption(f'Showing {len(view)} of {len(d)} customers')
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.download_button('⬇️ Download filtered data (CSV)', view.to_csv(index=False).encode('utf-8'), 'filtered_customer_segments.csv', 'text/csv')
else:
    st.subheader('Insights & recommendations')
    if d.empty: st.info('Choose at least one group in the sidebar.')
    else:
        profile = d.groupby('Cluster').agg(Customers=('CustomerID','count'), AvgAge=('Age','mean'), AvgIncome=('AnnualIncome','mean'), AvgSpend=('SpendingScore','mean')).round(1)
        income_avg, spend_avg = d.AnnualIncome.mean(), d.SpendingScore.mean()
        for cluster,row in profile.iterrows():
            if row.AvgIncome >= income_avg and row.AvgSpend >= spend_avg:
                label, advice = 'Higher income + higher spending', 'Consider loyalty rewards and relevant premium offers.'
            elif row.AvgSpend >= spend_avg:
                label, advice = 'Higher spending', 'Test relevant product recommendations and engagement campaigns.'
            elif row.AvgIncome >= income_avg:
                label, advice = 'Higher income + lower spending', 'Explore product fit and customer feedback before offering discounts.'
            else:
                label, advice = 'Lower income + lower spending', 'Test affordable bundles and introductory offers, then measure results.'
            with st.container(border=True):
                st.markdown(f'**Cluster {cluster} — {label}**')
                st.write(f'{int(row.Customers)} customers • Average age {row.AvgAge} • Average income {row.AvgIncome}')
                st.write(advice)
        st.info('Recommendations are hypotheses based on group averages, not guaranteed outcomes. Validate them with real customer research.')
st.divider()
st.caption('Built with Streamlit, Pandas and Plotly. Cluster numbers are labels, not rankings. Income units follow the supplied dataset.')
