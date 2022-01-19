# Library
# Data Analysis
import numpy as np
import pandas as pd

# Data Visualisation
import plotly.express as px
import folium as folium
# Maps
import geojson
import json

# Machine Learning Library
from sklearn import metrics
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder

# XGBoost
import xgboost as xgb
from xgboost import XGBRegressor
from warnings import filterwarnings

filterwarnings('ignore')

from dash import dcc, dash_table
from dash import html
import dash_bootstrap_components as dbc

# Colors dictionary for html
colors = {
    'background': '#111111',
    'title': '#7FDBFF',
    'text': '#a6a6a6',
    'blue': '#000067'
}

# Data Analysis
data = pd.read_csv("../resources/housing.csv", encoding='utf-8', sep=',')
data.drop_duplicates()

# Infinite Missing values

print("Percentage of missing values on each column :\n")
print(((data.isnull().sum() / data.shape[0]) * 100).sort_values(ascending=False))
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data = data.dropna()

# Boxplots
col_boxplots = data.columns.tolist()
col_boxplots.remove('ocean_proximity')
boxplots_fig = px.box(data,
                      y=col_boxplots,
                      title='Columns boxplot before cleanup',
                      log_y=True)

#################################
# DATAVIZ

# Cor Heatmap
cor_heatmap = px.imshow(data.corr(), color_continuous_scale='RdBu_r', text_auto=True)

# Pairplot / Scatter Matrix
pairplot_fig = px.scatter_matrix(data,
                                 dimensions=['housing_median_age', 'total_rooms', 'median_income',
                                             'median_house_value', 'ocean_proximity'],
                                 color="median_house_value",
                                 symbol="median_house_value",
                                 title="Scatter matrix of houses datasets",
                                 labels={col: col.replace('_', ' ') for col in data.columns})
pairplot_fig.update_traces(diagonal_visible=False)

# OP pie chart
OP_data = data.groupby(['ocean_proximity'], dropna=True).size().reset_index(name='count')
OP_pie_fig = px.pie(OP_data,
                    values='count',
                    names='ocean_proximity',
                    title='Ocean Proximity Distribution',
                    height=700,
                    width=1000
                    )

# Scatterplot : Median Income, House Value and Ocean Proximity
median_value_OP_scatter_fig = px.scatter(
    data,
    x="median_house_value",
    y="median_income",
    color='ocean_proximity',
    labels={'median_house_value': "Median House Value",
            'median_income': "Median Income",
            'ocean_proximity': "Ocean Proximity"
            },
    title="Median Income, House Value and Ocean Proximity scatterplot",
    height=800
)

# MAPS

# Population concentration depending on location
population_scatter_fig = px.scatter(
    data,
    x="longitude",
    y="latitude",
    hover_data=['population'],
    color="median_house_value",
    size=data["population"] / 100,
    labels={"population": 'Population'},
    title='Population concentration depending on location')

# Median House Value and other characteristics
all_houses_map = px.scatter_mapbox(data,
                                   lat='latitude',
                                   lon='longitude',
                                   color='median_house_value',
                                   hover_name='median_house_value',
                                   hover_data=['median_house_value', 'median_income',
                                               'population', 'total_rooms', 'total_bedrooms'],
                                   color_continuous_scale=px.colors.cyclical.IceFire,
                                   zoom=11,
                                   mapbox_style="open-street-map",
                                   width=1000,
                                   height=1000,
                                   title='Every house in California\'s map')

# LabelEncoding
le = LabelEncoder()
le_count = 0

for col in data:
    if data[col].dtype == 'object' or data[col].dtype == 'string':
        le.fit(data[col])
        data[col] = le.transform(data[col])
        le_count += 1
        print(col)
data.reset_index()
afterLE_string = str(le_count) + ' columns were label encoded.'
afterLE_string += "After the labelEncoding, ocean proximity has 5 categories :\n\+" \
                  "0 == '<1H OCEAN',\n\+" \
                  "1 == 'INLAND',\n\+" \
                  "2 == 'ISLAND',\n\+" \
                  "3 == 'NEAR BAY',\n\+" \
                  "4 == 'NEAR OCEAN'.\n"

######################################
# MACHINE LEARNING

evaluation_explaination = 'MAE takes the differences in all of the predicted and actual prices, adds them up and then divides them by the number of observations. It doesn’t matter if the prediction is higher or lower than the actual price, the algorithm just looks at the absolute value. A lower value indicates better accuracy.' \
                          'As a result of the squaring, MSE assigns more weight to the bigger errors. The algorithm then continues to add them up and average them. If you are worried about the outliers, this is the number to look at. Keep in mind, it’s not in the same unit as our dependent value. In our case, the value was roughly 82,3755,495, this is NOT the dollar value of the error like MAE. As before, lower the number the better.' \
                          'RMSE is the square root of MSE. This number is in the same unit as the value that was to be predicted. The value is usually higher than MAE.' \
                          'MSE and RMSE are really useful when you want to see if the outliers are messing with your predictions.' \
                          'Also known as the coefficient of determination, the r2_score works by measuring the amount of variance in the predictions explained by the dataset. Simply put, it is the difference between the samples in the dataset and the predictions made by the model.' \
                          'If the value of the r squared score is 1, it means that the model is perfect and if its value is 0, it means that the model will perform badly on an unseen dataset. This also implies that the closer the value of the r squared score is to 1, the more perfectly the model is trained.'

X = data[['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms',
          'population', 'households', 'median_income', 'ocean_proximity']]
y = data['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# XGBoost
bst = XGBRegressor()
bst.fit(X, y)

bst_pred = bst.predict(X_test)
predictions = "Predictions:\n\n" + str(bst_pred) + '\n'
real = "Real values:\n\n" + str(y_test)
MAE = "MAE: " + str(mean_absolute_error(y_test, bst_pred))
MSE = "MSE: " + str(mean_squared_error(y_test, bst_pred))
RMSE = "RMSE: " + str(np.sqrt(mean_squared_error(y_test, bst_pred)))
R2 = 'R2 Score: ' + str(r2_score(y_test, bst_pred))
R2_cross = "R2 Score using cross validation: " + str(cross_val_score(bst, X, y, cv=10, scoring="r2").mean())

######################################################

# Menu SideBar
navbar = dbc.NavbarSimple(
    [
        dbc.NavItem(dbc.NavLink("Home", href="/")),
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("More pages", header=True),
                dbc.DropdownMenuItem("Data Study & Cleanup", href="/data"),
                dbc.DropdownMenuItem("XGBoost", href="/xgboost"),
            ],
            nav=True,
            in_navbar=True,
            label="More",
        ),
    ],
    brand="Menu",
    brand_href="/",
    color="navbar-bran",
    dark=True,
)

url_bar_and_content_div = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content'),
])

# Home

layout_index = html.Div([
    # Title & Description
    html.H1('Housing Prices Prediction'),
    html.Hr(),
    html.P('In California, real estate agents need the help of machine learning models to determine housing prices. '
           'This is a Regression problem.'),
    html.Br(),

    # Summary
    html.H2('Summary'),
    dcc.Link('Data Study & cleanup', href='/data'),
    html.Br(),
    dcc.Link('XgBoost', href='/xgboost'),
    html.Br(),

    # Conclusion
    html.H3('Conclusion'),
    html.P('The best two models are Random Forest and XGboost with an R2 score respectively equal to 81% and 93%.'
           'However, our models seem to suffer from overfitting because their R2 score drops to around 54% '
           'when using cross validation ! This is why our models still need improvement.'),
    html.Br(),
])

# Data Study & Cleanup
layout_page_0 = html.Div([
    html.H1('Data Study & Cleanup'),

    # TODO ERROR
    #     html.Div(className='container',
    #              children=[
    #                  html.P('Ocean Proximity Distribution PieChart:'),
    #                  dcc.Graph(figure=OP_pie_fig),
    #              ], style={'textAlign': 'center'}),

    html.Div(className='container',
             children=[
                 html.P('Correlation Heatmap:'),
                 dcc.Graph(figure=cor_heatmap),
             ], style={'textAlign': 'center'}),

    html.Div(className='container',
             children=[
                 html.P('Pairplot:'),
                 dcc.Graph(figure=pairplot_fig),
             ], style={'textAlign': 'center'}),

    html.Div(className='container',
             children=[
                 html.P('Ocean Proximity Distribution PieChart:'),
                 dcc.Graph(figure=OP_pie_fig),
             ], style={'textAlign': 'center'}),

    html.Div(className='container',
             children=[
                 html.P('Median Income, House Value and Ocean Proximity Scatterplot:'),
                 dcc.Graph(figure=median_value_OP_scatter_fig),
             ], style={'textAlign': 'center'}),

    html.Div(className='container',
             children=[
                 html.P('Population concentration depending on location'),
                 dcc.Graph(figure=population_scatter_fig),
             ], style={'textAlign': 'center'}),

    html.Div(className='container',
             children=[
                 html.P('Median House Value and other characteristics on Map'),
                 dcc.Graph(figure=all_houses_map),
             ], style={'textAlign': 'center'}),
])

# XGBoost TODO
layout_page_1 = html.Div([
    html.H1('XGBoost')
])
