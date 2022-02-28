import numpy as np
from dash import Dash, dcc, html, Input, Output, no_update
import plotly.graph_objects as go
from PIL import Image
import io
import base64
import pickle
import math
import sys

app = Dash(__name__)

def np_gs_image_to_base64(im_matrix):
    if len(im_matrix.shape) == 1:
        side_length = int(math.sqrt(im_matrix.shape[0]))
        im_matrix = im_matrix.reshape((side_length, side_length))

    im = Image.fromarray(im_matrix)
    im = im.convert("RGB")
    buffer = io.BytesIO()
    im.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image
    return im_url

def initialize_plot(data, color):
    fig = go.Figure(data=[
    go.Scatter3d(
        x=data[:,0],
        y=data[:,1],
        z=data[:,2],
        mode="markers",
        marker=dict(size=1, color=color)
    )
    ])
    app.layout = html.Div([
        dcc.Graph(id="graph-basic-5", figure=fig, clear_on_unhover=True),
        dcc.Tooltip(id="graph-tooltip"),
    ])
    fig.update_traces(hoverinfo="none", hovertemplate=None)
    return fig

def visualize(npz_object):
    print("Identified the following keys in metadata: ", list(npz_object.keys()))
    data = npz_object["__data__"]
    color = npz_object["__color__"]
    fig = initialize_plot(data, color)

    data_map = npz_object["__data_map__"][()]
    metadata = {}
    for metadata_key in npz_object:
        if metadata_key == '__data__' or metadata_key == '__data_map__' or metadata_key == '__color__':
            pass
        else:
            dtype = data_map[metadata_key]
            metadata[metadata_key] = (dtype, npz_object[metadata_key])

    @app.callback(
        Output("graph-tooltip", "show"),
        Output("graph-tooltip", "bbox"),
        Output("graph-tooltip", "children"),
        Input("graph-basic-5", "hoverData"),
    )
    def display_hover(hoverData):
        if hoverData is None:
            return False, no_update, no_update

        pt = hoverData["points"][0]
        bbox = pt["bbox"]
        num = pt["pointNumber"]
        
        html_element_list = []
        for metadata_key in metadata:
            dtype, d = metadata[metadata_key]
            if dtype == '__greyscale_image__':
                img = np_gs_image_to_base64(d[num])
                html_element_list.append(html.Img(src=img, style={"width": "100%"}))
            elif dtype == '__scalar__':
                html_element_list.append(html.P("{}: {}".format(metadata_key, d[num])))
            elif dtype == '__vector__':
                html_element_list.append(html.P("{}: {}".format(metadata_key, d[num]))) #TODO: Make this nicer
            else:
                raise ValueError("Unsupported data type: {}".format(dtype))
            
        children = [
            html.Div(html_element_list, style={'width': '200px', 'display': 'block', 'margin': '0 auto'})
        ]

        return True, bbox, children
    app.run_server(debug=True)


if __name__=='__main__':
    visualize(np.load(sys.argv[1], allow_pickle=True))
