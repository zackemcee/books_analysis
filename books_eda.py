# # This dashboard has the purpose for data visualization of the books dataset based on certain columns.
# # It serves as a tool to do some grouping and EDA.
# # Initially, it was going to be hosted on Heroku, but since the service is no longer free, it is going to be hosted on Dash's dashboard renderer. 

# Main imports
import pandas as pd
import numpy as np
import plotly.express as px
import os
# Dash for the dashboard
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import dash_table as dt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import warnings
warnings.filterwarnings("ignore")


# Main dataframe
df = pd.read_csv(
    r'https://raw.githubusercontent.com/zackemcee/books_analysis/main/books.csv', on_bad_lines='skip')
df.columns = [i.replace(' ', '') for i in df.columns]
df['year'] = df['publication_date'].str[-4:].astype(int)
df = df[['publisher', 'authors', 'language_code', 'publication_date'] +
        [i for i in df.columns.drop(['publisher', 'authors', 'language_code', 'isbn', 'isbn13', 'publication_date'])]]

# Main app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div([
    # Title
    html.H1('GoodReads EDA Dashboard', style={
            'width': '100%', 'text-align': 'center', 'font-weight': 'bold', 'margin': '25px 0 5px 0'}),

    # First segment
    html.Div([
        # Filter by
        html.H4('Filter by', style={
                'width': '100%', 'font-weight': 'bold', 'text-align': 'center'}),
        html.Div(dcc.RadioItems(['publisher', 'authors', 'year','language_code'], 'authors', id='radio'),
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


# App callback
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
    df1 = df1[(df1.bookID >= value2) & (df1.bookID <= maxed) & (df1.average_rating >= value3)] # ---------------- To verify
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
    app.run_server(debug=True)
