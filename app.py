from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import io
import base64
import plotly.graph_objects as go

app = Flask(__name__)

#Load pickle file in the data frame 
Districts_cluters_withAllFeature = pd.read_pickle('Districts_clusters.pkl')
df = pd.read_pickle('Rajasthan_data_cleaned.pkl')
#df = pd.read_csv('D:\MGNREGA_WEBSITE\MGNREGA_RAJASTHAN_JUNE_ANALYSIS\MGNREGA_DONE.csv')

# Define the desired values of K for each feature
k_values = {
    'Household': 2,
    'Area': 2,
    'Population': 3,
    'SCs_RW': 3,
    'STs_RW': 2,
    'Others_RW': 2,
    'Women_RW': 2,
    'Persons': 2,
    'SCs_AW': 2,
    'STs_AW': 2,
    'Others_AW': 2,
    'Women_AW': 2,
    'Literacy': 2
}

def get_plot_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return plot_base64



# Define routes and views
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/conclusion')
def conclusion():
    return render_template('conclusion.html')

@app.route('/Description')
def discription():
    return render_template('Description.html')

@app.route('/eda')
def eda():
        return render_template('eda.html')
    
@app.route('/heatmap')
def heatmap():
         corr_matrix = df.corr(numeric_only=True)

    # Create the heatmap trace
         trace = go.Heatmap(
              z=corr_matrix.values,
              x=corr_matrix.columns,
              y=corr_matrix.index,
             colorscale='inferno',  # You can choose other color scales as well
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

         return render_template('heatmap.html', plot_div=plot_div)     


@app.route('/cluster_graph') 
def cluster_graph():
    return render_template('cluster_graph.html')  

@app.route('/clusters', methods=['GET', 'POST'])
def clusters():
    if request.method == 'POST':
        cluster_id = int(request.form['cluster_id'])
        #filter the clusters based on the selected cluster_id
        filtered_districts = Districts_cluters_withAllFeature[Districts_cluters_withAllFeature['Clusters'] == cluster_id]
        districts_list = filtered_districts['District'].tolist()
        return render_template('clusters.html', districts=districts_list)
    return render_template('clusters_input.html')

@app.route('/clustering', methods=['GET', 'POST'])
def clustering():
    if request.method == 'POST':
        feature_input = request.form['feature_input']
        # check if input is valid
        if feature_input in df.columns:
            # get the selected feature and 'District' column from the dataframe
            selected_df = df[[feature_input, 'District']].copy()

            # get desired value of k for selected feature
            k = k_values.get(feature_input)

            if k is not None:
                # apply k means clustering
                kmeans = KMeans(n_clusters=k, random_state=42)
                clusters = kmeans.fit_predict(selected_df[[feature_input]])

                # adding 1 to make cluster value from 0 to 1
                clusters += 1

                # Add the cluster assignments to the selected dataframe
                selected_df.loc[:, 'Cluster'] = clusters

                # group the district by clusters
                grouped_df = selected_df.groupby('Cluster')['District'].apply(list)

                # Create a bar graph using Plotly
                fig = go.Figure()
                for i, districts in grouped_df.items():
                    fig.add_trace(go.Bar(x=districts, y=[i] * len(districts), name=f'Cluster {i}'))

                fig.update_layout(
                    xaxis_title='District',
                    yaxis_title='Cluster',
                    title='Districts Belonging to Clusters',
                    xaxis_tickangle=-45,
                    bargap=0.2
                )

                # Convert the plot to a div string to be used in the template
                plot_div = fig.to_html(full_html=False)

                return render_template('clustering.html', plot_div=plot_div)
            else:
                return "K value is not defined for the selected feature."
        else:
            return "Invalid feature name. Please try again."
    else:
        return render_template('clustering.html')   





if __name__ == '__main__':
    app.run(debug=True)
