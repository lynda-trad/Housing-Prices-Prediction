from dash import dcc
from dash import html
from dash.dependencies import Input, Output

from dash_deploy.app import app
from layouts import navbar, layout_index, \
    layout_page_0, layout_page_1

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == "/data":
        return layout_page_0
    elif pathname == "/xgboost":
        return layout_page_1
    else:
        return layout_index


if __name__ == '__main__':
    app.run_server(debug=True)
