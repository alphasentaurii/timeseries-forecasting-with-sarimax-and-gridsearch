# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import pandas as pd
# PLOTLY / CUFFLINKS for iplots
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cufflinks as cf
cf.go_offline()
import json
import statspack as spak

#external_stylesheets = ['style.css']
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#external_stylesheets = [dbc.themes.CYBORG]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    'background': '#ffffff',#'#111111', 
    'text': '#000000' #'#7FDBFF'
}
#colors = ["royalblue","crimson","lightseagreen","orange","lightgrey"]

# LOAD DATA
data = pd.read_csv('data/newyork_1996-2018.csv', parse_dates=True)
NY = pd.DataFrame(data)
df = spak.makeTime(NY, idx='DateTime')


available_zipcodes = df['RegionName'].unique()
# Create dicts
# NYC: dict of cities and zip codes
# nyc: dict of dataframes for each zip code
NYC, nyc, city_zip = spak.cityzip_dicts(df=df, col1='RegionName', col2='City')


####### Forecast Prediction Values
df_preds = pd.read_csv('data/ny_predictions.csv', parse_dates=True)
df_preds = df_preds.drop('RegionID', axis=1)
df_preds = spak.makeTime(df_preds, idx='DateTime')


# FC = spak.makeTime(df_preds, idx='DateTime')
# FC = pd.DataFrame()
FC = df_preds.loc[df_preds.index > '2018-04-01']

# Train Lines
NY_Newhaven=pd.read_csv('data/newhaven.csv')
NY_Harlem=pd.read_csv('data/harlem.csv')
NY_Hudson=pd.read_csv('data/hudson.csv')

#     if start is None:
#         start = test.index[0]     
#     if end is None:
#         end = test.index[-1]    
        
#     # Get predictions starting from 2013 and calculate confidence intervals.
#     prediction = model_output.get_prediction(start=start,end=end, dynamic=True)
    
#     forecast = prediction.conf_int()
#     forecast['predicted_mean'] = prediction.predicted_mean
#     fc_plot = pd.concat([forecast, train], axis=1)


# FIGURES

# fig = df.iplot(kind='bar', x='Month', y='MeanValue', title='Time Series with Range Slider and Selectors', asFigure=True)

# fig.update_xaxes(
#     rangeslider_visible=True,
#     rangeselector=dict(
#         buttons=list([
#             dict(count=1, label="1m", step="month", stepmode="backward"),
#             dict(count=6, label="6m", step="month", stepmode="backward"),
#             dict(count=1, label="YTD", step="year", stepmode="todate"),
#             dict(count=1, label="1y", step="year", stepmode="backward"),
#             dict(step="all")
#         ])
#     )
# )

# fig1

fig1 = go.Figure()
# fig1.add_trace(go.Scatter(x=df.index, y=df['MeanValue'], name="Mean Home Value",line_color='crimson'))
# fig1.add_trace(go.Scatter(x=FC.index, y=FC['pred_mean'], name="Forecast Value",line_color='deepskyblue'))
# fig1.add_trace(go.Scatter(x=NY_Hudson['DateTime'], y=NY_Hudson['MeanValue'], name="Hudson MeanValue",
#                          line_color='lightgreen'))
# fig1.update_layout(title_text='MeanValues by Train Line',
#                   xaxis_rangeslider_visible=True)

txd = spak.time_dict(d=NYC, xcol='RegionName', ycol='MeanValue')
for k,v in txd.items():
    fig1.add_trace(go.Line(x=NY['Month'].loc[NY['RegionName']==k], y=NY['MeanValue'].loc[NY['RegionName']==k], name=str(k)))

fig1.update_layout(title_text='Westchester County NY - Mean Home Values',
                  xaxis_rangeslider_visible=True)


fig1.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)



#### FIG 3

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=NY_Newhaven['DateTime'], y=NY_Newhaven['MeanValue'], name="NewHaven MeanValue",
                         line_color='crimson'))
fig3.add_trace(go.Scatter(x=NY_Harlem['DateTime'], y=NY_Harlem['MeanValue'], name="Harlem MeanValue",
                         line_color='deepskyblue'))
fig3.add_trace(go.Scatter(x=NY_Hudson['DateTime'], y=NY_Hudson['MeanValue'], name="Hudson MeanValue",
                         line_color='lightgreen'))
fig3.update_layout(title_text='MeanValues by Train Line',
                  xaxis_rangeslider_visible=True)


top5 = df.loc[(df['RegionName'] == 10708) | (df['RegionName']==10706) | (df['RegionName']==10803) | (df['RegionName']==10514) | (df['RegionName']==10605) ]

fig4 = go.Figure()
fig4 = px.scatter(top5, x='Month', y='MeanValue')
#fig4.add_trace(go.Scatter(x=df['DateTime']['10708'], y=df['MeanValue']['10708'], name="10708"))
fig4.update_layout(title_text='Top 5 Zip Codes',
                  xaxis_rangeslider_visible=True)

#top5 = df.loc[df['RegionName']  = ['10708', '10706', '10803', '10514', '10605'])
#fig4 = px.scatter(top5, x='Month', y='MeanValue', name='RegionName')
#fig4.update_layout(title_text='Top 5 Zip Code Mean Values',
                  #xaxis_rangeslider_visible=True)
#mapTime(NYC,xcol='RegionName', ycol='MeanValue',X=top5recs, vlines=True, MEAN=True)


###

# fig = px.density_heatmap(df, x="ROI", y="conf_mean", marginal_x="rug", marginal_y="histogram")
# fig.show()


# ts_data = go.Scatter(x=df.index,y=df.MeanValue)
# layout = go.Layout(title='Timeseries Plot', xaxis=dict(title='Date'),yaxis=dict(title='(Price)'))
# fig3 = go.Figure(data=[ts_data], layout=layout)
# pyo.iplot(fig3, sharing='public')

# fig.update_layout(
#     plot_bgcolor=colors['background'],
#     paper_bgcolor=colors['background'],
#     font_color=colors['text']
# )

# fig,ax = mapTime(d=NYC, xcol='RegionName', ycol='MeanValue', X=[], vlines=True, MEAN=True)

# fig3 = go.Figure()

# fig3 = plt.figure(figsize=(13,5))
#     ts1.plot(label='d=1',figsize=(13,5), c='blue',lw=1,alpha=.7)
#     ts2.plot(label='d=2',figsize=(13,5), c='red',lw=1.2,alpha=.8)
#     ts3.plot(label='d=3',figsize=(13,5), c='magenta',lw=1,alpha=.7)
#     ts4.plot(label='d=4',figsize=(13,5), c='green',lw=1,alpha=.7)
#     plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=True, 
#                fancybox=True, facecolor='lightgray')
#     plt.tight_layout()
#     plt.gcf().autofmt_xdate()
#     plt.show();


# DATA TABLES

def generate_table(dataframe, max_rows=5):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ]), 
    ])

# FORECASTING
# r,forecast, fig, ax = forecastX(model_output, train, test, get_metrics=True)
# gridX, best_params = gridMAX(ts,pdq=pdq)
# metagrid, ROI = gridMAXmeta(KEYS=NYC, s=False)

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='RealtyRabbit',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),

    html.H4(
        children='A real estate forecasting application for home buyers.', 
        style={
        'textAlign': 'center',
        'color': colors['text']
        }
    ),

    # dcc.Graph(
    #     id='timeseries',
    #     figure=fig
    # ),

    dcc.Graph(
        id='ts',
        figure=fig1
    ),

    # dcc.Input(
    #     id='number-in',
    #     value=10701,
    #     style={'fontSize':28}
    # ),
    # html.Button(
    #     id='submit-button',
    #     n_clicks=0,
    #     children='Submit',
    #     style={'fontSize':28}
    # ),
    # html.H1(id='number-out'),

    dcc.Graph(
        id='ts_trainlines',
        figure=fig3
    ),

    dcc.Graph(
        id='top5-zipcodes',
        figure=fig4
    ),

    generate_table(df_preds),

    
    dcc.Graph(
        id='clientside-graph'
    ),
    dcc.Store(
        id='clientside-figure-store',
        data=[{
            'x': df_preds[df_preds['RegionName'] == '10701'].index,
            'y': df_preds[df_preds['RegionName'] == '10701']['MeanValue']
        }]
    ),
    'Indicator',
    dcc.Dropdown(
        id='clientside-graph-indicator',
        options=[
            {'label': 'Mean Value', 'value': 'MeanValue'},
            # {'label': 'SizeRank', 'value': 'SizeRank'},
            # {'label': 'Rolling Average', 'value': 'RollingAvg'}
        ], 
        value='MeanValue'
    ),
    'RegionName',
    dcc.Dropdown(
        id='clientside-graph-zipcode',
        options=[
            {'label': RegionName, 'value': RegionName}
            for RegionName in available_zipcodes
        ],
        value='10701'
    ),
    'Graph scale',
    dcc.RadioItems(
        id='clientside-graph-scale',
        options=[
            {'label': x, 'value': x} for x in ['linear', 'log']
        ],
        value='linear'
    ),
    html.Hr(),
    html.Details([
        html.Summary('Contents of figure storage'),
        dcc.Markdown(
            id='clientside-figure-json'
        )
    ])

])

# @app.callback(
#     Output('number-out', 'children'),
#     [Input('submit-button', 'n_clicks')],
#     [State('number-in', 'value')])
# def output(n_clicks, number):
#     return '{} displayed after {} clicks'.format(number,n_clicks)


@app.callback(
    Output('clientside-figure-store', 'data'),
    [Input('clientside-graph-indicator', 'value'),
     Input('clientside-graph-zipcode', 'value')]
)
def update_store_data(indicator, zipcode):
    dff = df_preds[df_preds['RegionName'] == zipcode]
    return [{
        'x': dff.index,
        'y': dff[indicator],
        'mode': 'markers'
    }]


app.clientside_callback(
    """
    function(data, scale) {
        return {
            'data': data,
            'layout': {
                 'yaxis': {'type': scale}
             }
        }
    }
    """,
    Output('clientside-graph', 'figure'),
    [Input('clientside-figure-store', 'data'),
     Input('clientside-graph-scale', 'value')]
)


@app.callback(
    Output('clientside-figure-json', 'children'),
    [Input('clientside-figure-store', 'data')]
)
def generated_figure_json(data):
    return '```\n'+json.dumps(data, indent=2)+'\n```'


if __name__ == '__main__':
    app.run_server(debug=True,host='127.0.0.1')
