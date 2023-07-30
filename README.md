# MGNREGA_RAJASTHAN_JUNE_ANALYSIS
This website present the data analysis done on state Rajasthan  
import plotly.graph_objects as go
import pandas as pd
from flask import Flask, request, render_template
from sklearn.cluster import KMeans

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from flask import Flask, render_template



@app.route('/correlation_heatmap', methods=['GET'])
def correlation_heatmap():
    corr_matrix = df.corr(numeric_only=True)

    # Create the heatmap trace
    trace = go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='Viridis',  # You can choose other color scales as well
        colorbar=dict(title='Correlation')
    )

    # Create the layout
    layout = go.Layout(
        title='Correlation Heatmap',
        xaxis=dict(title='Features'),
        yaxis=dict(title='Features')
    )

    # Create the figure
    fig = go.Figure(data=trace, layout=layout)

    # Convert the plot to a div string
    plot_div = fig.to_html(full_html=False)

    return render_template('correlation_heatmap.html', plot_div=plot_div)


if __name__ == '__main__':
    app.run(debug=True)

