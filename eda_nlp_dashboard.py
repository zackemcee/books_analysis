# # This dashboard has the purpose for data visualization of the books dataset based on certain columns.
# # It serves as a tool to do some grouping and EDA.
# # Initially, it was going to be hosted on Heroku, but since the service is no longer free, it is going to be hosted on Dash's dashboard renderer.

# Main imports
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
# Dash for the dashboard
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import dash_table as dt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
# Sklearn for label mapping and predictions
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Main dataframe
df = pd.read_csv(
    r'https://raw.githubusercontent.com/zackemcee/books_analysis/main/books.csv', on_bad_lines='skip')
df.columns = [i.replace(' ', '') for i in df.columns]
df['year'] = df['publication_date'].str[-4:].astype(int)
df = df[['publisher', 'authors', 'language_code', 'publication_date'] +
        [i for i in df.columns.drop(['publisher', 'authors', 'language_code', 'isbn', 'isbn13', 'publication_date'])]]
df2 = df.copy().query(
    'num_pages <= 1000 & ratings_count <= 500000 & text_reviews_count <= 5000')
book_encoder = LabelEncoder()
author_encoder = LabelEncoder()
lge_encoder = LabelEncoder()
df2['book'] = book_encoder.fit_transform(df2.title)
df2['author'] = author_encoder.fit_transform(df2.authors)
df2['lge'] = lge_encoder.fit_transform(df2.language_code)
df2['year'] = df2['publication_date'].str[-4:].astype(int)
# # Main models
# Linear Regression
lr = pickle.load(open('lr_model.sav', 'rb'))
# Random Forest Regressor
rfr = pickle.load(open('rf_model.sav', 'rb'))

# Main app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div([
    # Title Prediction/ML
    html.H1('GoodReads ML Predictions', style={
            'width': '100%', 'text-align': 'center', 'font-weight': 'bold', 'margin': '25px 0 5px 0'}),
    # Filter segment
    html.Div([
        # Filter by
        html.H4('Choices', style={
                'width': '70%', 'font-weight': 'bold', 'text-align': 'center', 'margin': 'auto'}),
        html.Div([
            # Title, author, year and language
            dcc.Dropdown(df2.title.sort_values(ascending=True).unique(),
                         np.random.choice(df2.title), id='book', style={'width': '400px', 'margin': '10px'}),
            dcc.Dropdown(df2.authors.sort_values(ascending=True).unique(),
                         np.random.choice(df2.authors), id='author', style={'width': '400px', 'margin': '10px'}),
            dcc.Dropdown(df2.year.sort_values(ascending=False).unique(),value=2005,
                         id='year', placeholder='Select a year: ', style={'width': '400px', 'margin': '10px'}),
            dcc.Dropdown(df.language_code.sort_values(ascending=True).unique(), 
                         np.random.choice(df2.language_code), id='language', placeholder='Select a language: ', style={'width': '400px', 'margin': '10px'}),
            # Number of pages slider
            html.H6('Select a number of pages: ', style={
                    'text-align': 'center', 'color': 'white', 'width': '100%'}),
            html.Div([dcc.Slider(df2.num_pages.min(), df2.num_pages.max(), value=df2.num_pages.sample().iloc[0], id='n_pages', marks=None,
                     step=1, tooltip={"placement": "bottom", "always_visible": True})], style={'width': '100%', 'margin': '0 auto'}),
            # Number of ratings slider
            html.H6('Select a number of ratings: ', style={
                    'text-align': 'center', 'color': 'white', 'width': '100%'}),
            html.Div([dcc.Slider(df2.ratings_count.min(), df2.ratings_count.max(), value=df2.ratings_count.sample().iloc[0], id='rating_count',
                     step=1, marks=None, tooltip={"placement": "bottom", "always_visible": True})], style={'width': '100%', 'margin': '0 auto'}),
            # Number of text reviews slider
            html.H6('Select a number of text reviews: ', style={
                    'text-align': 'center', 'color': 'white', 'width': '100%'}),
            html.Div([dcc.Slider(df2.text_reviews_count.min(), df2.text_reviews_count.max(), value=df2.text_reviews_count.sample(
            ).iloc[0], id='text_review_count', step=1, marks=None, tooltip={"placement": "bottom", "always_visible": True})], style={'width': '100%', 'margin': '0 auto'}),
        ], style={'width': '70%', 'color': 'black', 'margin': '10px', 'display': 'flex', 'justify-content': 'center', 'flex-wrap': 'wrap'}),
        dcc.Dropdown(['Linear Regression', 'Random Forest Regressor'], placeholder='Select a Machine Learning model:',
                     id='model_selection', style={'color': 'black', 'width': '60%', 'margin': '10px auto'}),

        # Text & predictions
        html.Div([
            html.Div([html.H6('Predicted rating:'),
                      html.Div(id='dd-output-container2')],
                     style={'text-align': 'center', 'margin': 'auto'}),
        ], style={'display': 'flex', 'flex-wrap': 'wrap', 'width': '50%', 'margin': '0 0 10px 0', 'background-color': 'white', 'color': 'black', 'border-radius': '10px'})

    ], style={'width': '98%', 'display': 'flex', 'justify-content': 'space-evenly', 'flex-wrap': 'wrap', 'background-color': '#061E44', 'margin': '10px auto'}),

    # Title EDA
    html.H1('GoodReads EDA Dashboard', style={
            'width': '100%', 'text-align': 'center', 'font-weight': 'bold', 'margin': '25px 0 5px 0'}),
    # Filter segment
    html.Div([
        # Filter by
        html.H4('Filter by', style={
                'width': '100%', 'font-weight': 'bold', 'text-align': 'center'}),
        html.Div(dcc.RadioItems(['publisher', 'authors', 'year', 'language_code'], 'authors', id='radio'),
                 style={'width': '80%', 'text-align': 'center', 'text-transform': 'capitalize', 'font-weight': 'bold', 'margin': '0', 'text-wrap': 'wrap', 'display': 'flex', 'justify-content': 'center'}),

        # Slider
        html.Div([
            html.H4('Filter by',
                    style={'width': '100%', 'font-weight': 'bold', 'text-align': 'center', 'margin': '20px 0 0 0'}),
            html.Div([
                html.Div([
                    html.H6('Number of Books', style={
                            'width': '100%', 'text-align': 'center', 'font-weight': 'bold'}),
                    html.Div(dcc.Slider(10, 150, 5, value=20,
                                        id='slider'), id='slider-div', style={'width': '100%', 'margin': '20px'}),
                ], style={'text-align': 'center', 'width': '700px', 'margin': '0 10px', 'display': 'flex', 'flex-wrap': 'wrap'}),
                html.Div([
                    html.H6('Average Rating', style={
                            'width': '100%', 'text-align': 'center', 'font-weight': 'bold'}),
                    html.Div(dcc.Slider(0, 5, 0.5, value=2,
                                        id='slider-2'), style={'width': '100%', 'margin': '20px'}),
                ], style={'text-align': 'center', 'width': '400px', 'margin': '0 10px', 'display': 'flex', 'flex-wrap': 'wrap'}),
            ], style={'width': '100%', 'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'center'}),
        ], style={'display': 'flex', 'justify-content': 'center', 'flex-wrap': 'wrap'})

    ], style={'width': '95%', 'display': 'flex', 'justify-content': 'space-evenly', 'flex-wrap': 'wrap', 'background-color': '#061E44'}
    ),

    # Main graph & table
    html.Div([
        # Graph
        dcc.Graph(id="graph", style={
            'width': '800px', 'height': '100%', 'padding': '20px'}),

        # Table
        html.Div([
            html.H4('Main Dataframe Table', style={
                    'text-align': 'center', 'width': '750px', 'padding': '5px'}),
            html.Div(id="table1", style={
                     'width': '750px', 'margin': '25px auto'}),
        ], style={'width': '800px', 'display': 'flex', 'flex-wrap': 'wrap', 'background-color': '#061E44', 'margin': '20px', 'height': '90%'})

    ], style={'width': '95%', 'height': '70%', 'display': 'flex', 'justify-content': 'space-evenly', 'flex-wrap': 'wrap'}),

    # Graph subplots
    html.Div([
        # Graph
        dcc.Graph(id="graph2", style={
            'width': '95%', 'height': '90%', 'padding': '20px', 'color': '#FFFFFF'}),

    ], style={'width': '95%', 'height': '70%', 'display': 'flex', 'justify-content': 'center', 'flex-wrap': 'wrap'}),



], style={'display': 'flex', 'width': '100%', 'height': '100%', 'margin': '0', 'padding': '0', 'justify-content': 'center', 'flex-wrap': 'wrap', 'background-color': '#0e103d', 'color': '#FFFFFF', 'font-family': 'Arial', 'line-height': '1.5', 'text-wrap': 'wrap'})


# ML App callback
@app.callback(
    Output('dd-output-container2', 'children'),
    Input('book', 'value'),
    Input('author', 'value'),
    Input('language', 'value'),
    Input('n_pages', 'value'),
    Input('rating_count', 'value'),
    Input('text_review_count', 'value'),
    Input('year', 'value'),
    Input('model_selection', 'value'),
)
def update_output(book, author, language, n_pages, rating_count, text_review_count, year, model):
    if author == None or book == None or language == None or model == None:
        prediction = html.H4(f'Select a model',
                             style={'font-weight': 'bold', 'text-align': 'center', 'margin': '0 auto', 'color': '#ff0f0f'}),
    else:
        book_val = book_encoder.transform([book])
        author_val = author_encoder.transform([author])
        language_val = lge_encoder.transform([language])
        to_pred = [[book_val, author_val, language_val,
                    n_pages, rating_count, text_review_count, year]]
        if model == 'Linear Regression':
            pred = lr.predict(to_pred)
            prediction = html.H4(f'{pred[0]}',
                                 style={'font-weight': 'bold', 'text-align': 'center', 'margin': '0 auto'}),
        else:
            pred = rfr.predict(to_pred)
            prediction = html.H4(f'{pred[0]}',
                                 style={'font-weight': 'bold', 'text-align': 'center', 'margin': '0 auto'}),
    return prediction

# EDA App callback


@app.callback(
    Output('graph', 'figure'),
    Output('table1', 'children'),
    Output('slider-div', 'children'),
    Output('graph2', 'figure'),
    Input('radio', 'value'),
    Input('slider', 'value'),
    Input('slider-2', 'value'),
)
def update_output(value1, value2, value3):
    df1 = df.groupby(value1).agg({'average_rating': 'mean', 'bookID': 'count', 'num_pages': 'mean',
                                 'ratings_count': 'sum', 'text_reviews_count': 'sum'}).reset_index().sort_values('bookID', ascending=False)

    # # Adjust the filters
    # max_value = value2
    if value1 == 'publisher':
        val = 25
        maxed = 150
    elif value1 == 'year' or value1 == 'language_code':
        val = 100
        maxed = 500
    else:
        val = 5
        maxed = 35

    df1['average_rating'] = df.average_rating.round(2)
    df1 = df1[(df1.bookID >= value2) & (df1.bookID <= maxed) & (
        df1.average_rating >= value3)]  # ---------------- To verify
    fig = px.bar_polar(df1, r="bookID", theta=value1,
                       hover_name=value1, hover_data={
                           'bookID': True, 'average_rating': True, value1: False},
                       labels={
                           "average_rating": "Average Rating",
                           "bookID": 'N° of Books',
                       },
                       color="average_rating", template='none',
                       color_continuous_scale=px.colors.sequential.dense_r)

    fig.update_layout(
        paper_bgcolor="#061E44",
        height=550,
        margin=dict(t=100),
        title={
            'text': f'<b>Rating based on {value1.upper()} with N° of books >= {value2} and an average rating >= {value3}<br>{len(df1)} books in total<br>',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        template='plotly_dark'
    )

    # Filter adjustment for slider
    slider = dcc.Slider(0, maxed, val, value=value2,
                        id='slider')
    # Fig 2
    fig2 = make_subplots(rows=1, cols=3, start_cell="bottom-left",
                         subplot_titles=("Average Rating", "Number of Pages", "Number of Books"))
    fig2.add_trace(go.Bar(x=df1[value1], y=df1.sort_values(
        'average_rating', ascending=False).average_rating, hovertemplate='<b>%{x}</b>: %{y}<extra></extra>'), row=1, col=1)
    fig2.add_trace(go.Bar(x=df1[value1], y=df1.sort_values(
        'num_pages', ascending=False).num_pages, hovertemplate='<b>%{x}</b>: %{y}<extra></extra>'), row=1, col=2)
    fig2.add_trace(go.Bar(x=df1[value1], y=df1.sort_values(
        'bookID', ascending=False).bookID, hovertemplate='<b>%{x}</b>: %{y}<extra></extra>'), row=1, col=3)
    fig2.update_layout(height=600, paper_bgcolor="#061E44",
                       template='plotly_dark', showlegend=False,
                       title={
                           'text': f'<b>{value1.capitalize()} by values</b>',
                           'y': 0.95,
                           'x': 0.5,
                           'xanchor': 'center',
                           'yanchor': 'top'},)
    # Table
    df1 = df1.rename(columns={'bookID': 'N° of Books', 'average_rating': 'Average Rating', 'authors': 'Author',
                     'num_pages': 'N° of Pages', 'ratings_count': 'N° of Ratings', 'text_reviews_count': 'N° of Text Reviews'})
    data = df1.to_dict('rows')
    columns = [{"name": i, "id": i, } for i in (df1.columns)]
    table = dt.DataTable(data=data, columns=columns,
                         filter_action='native',
                         sort_action='native',
                         page_size=11,
                         style_data={
                             'color': '#FFFFFF',
                             'backgroundColor': '#061E44',
                             'whiteSpace': 'normal',
                             'height': 'auto',
                         },
                         style_header={
                             'color': '#FFFFFF',
                             'backgroundColor': '#061E44'
                         },
                         style_filter={
                             'color': '#FFFFFF',
                             'backgroundColor': '#061E44'
                         })

    return fig, table, slider, fig2


if __name__ == '__main__':
    app.run_server(debug=False)
