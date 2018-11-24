
#### Network Analysis <br/>
#### (Team Project - Samir, Pablo, Amaya, Sarah and Arianne - MSBA UC Irvine 2018-19


```python
import pandas as pd
import numpy as np

df=pd.DataFrame(pd.read_csv('products.csv'))
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>title</th>
      <th>group</th>
      <th>salesrank</th>
      <th>review_cnt</th>
      <th>downloads</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Patterns of Preaching: A Sermon Sampler</td>
      <td>Book</td>
      <td>396585.0</td>
      <td>2</td>
      <td>2</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Candlemas: Feast of Flames</td>
      <td>Book</td>
      <td>168596.0</td>
      <td>12</td>
      <td>12</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>World War II Allied Fighter Planes Trading Cards</td>
      <td>Book</td>
      <td>1270652.0</td>
      <td>1</td>
      <td>1</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Life Application Bible Commentary: 1 and 2 Tim...</td>
      <td>Book</td>
      <td>631289.0</td>
      <td>1</td>
      <td>1</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Prayers That Avail Much for Business: Executive</td>
      <td>Book</td>
      <td>455160.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_filter=df[(df['group']=='Book')&((df['salesrank']<=150000)&(df['salesrank']>-1))]

df_cop=pd.DataFrame(pd.read_csv('copurchase.csv'))
df_cop_book=df_cop[df_cop.Source.isin(df_filter.id) & df_cop.Target.isin(df_filter.id)]
in_degree=df_cop_book.groupby(['Target'])['Source'].size().reset_index(name='in_degree')
out_degree=df_cop_book.groupby(['Source'])['Target'].size().reset_index(name='out_degree')

x = out_degree.set_index('Source')
y = in_degree.set_index('Target').rename_axis('Source')
y.columns = x.columns

combined=y.add(x, fill_value=0).loc[y.index, :].reset_index()
df_filter.head()
combined.nlargest(5,'out_degree') #Top 5 products with highest in+out degree
#2 products (33 and 4429) with highest degree.
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Source</th>
      <th>out_degree</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>33</td>
      <td>53.0</td>
    </tr>
    <tr>
      <th>360</th>
      <td>4429</td>
      <td>53.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>244</td>
      <td>36.0</td>
    </tr>
    <tr>
      <th>32</th>
      <td>302</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>471</th>
      <td>5913</td>
      <td>22.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#grouping by source and target products to see if any particular combination has been repeated. 
#We'll use this dataset to build the network graph
df_final_group=df_cop_book.groupby(['Source','Target']).size().reset_index(name='Freq')
df_final_group.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Source</th>
      <th>Target</th>
      <th>Freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12</td>
      <td>261</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>74</td>
      <td>282</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>77</td>
      <td>422</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>79</td>
      <td>82</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>117</td>
      <td>131</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



The product with the highest degree (in degree + out degree) are 33 and 4429. We’re interested in the sub component off all the products that are directly or indirectly associated with products 33 and 4429. The nodes 33 and 4429 and all its subcomponents were visualized by using a package Networkx. In the graph, larger the size of node, larger the degree for the node and darker color means larger degree. The degree of all nodes varies from 1-53. (A clear picture is attached with the assignment).


```python
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pylab
import community
from matplotlib.pyplot import subplots
%matplotlib inline

g = nx.from_pandas_edgelist(df_final_group, 'Source', 'Target', ['Freq'])#create_using=nx.Graph()

fig, ax = subplots()
subnet = nx.node_connected_component(g, 4429)
pos=nx.kamada_kawai_layout(g.subgraph(subnet))

cmapC = plt.cm.get_cmap('Spectral')
degrees = dict(g.subgraph(subnet).degree()) #Dict with Node ID, Degree

nodes = dict(g.subgraph(subnet).nodes())
n_color = np.asarray([degrees[n] for n in nodes])
edges = dict(g.subgraph(subnet).edges())
weights = [g.subgraph(subnet)[u][v]['Freq'] for u,v in edges]
colors=range(53)
vmin = min(colors)
vmax = max(colors)

draw=nx.draw_kamada_kawai(g.subgraph(subnet),k=1.2, with_labels = False,
               nodelist=degrees.keys(),node_size=[v*50 for v in degrees.values()]
                     ,cmap=cmapC,width=weights,arrows=True,node_color=n_color,vmin=vmin, vmax=vmax)

plt.xticks([], [])
plt.yticks([], [])
fig = plt.gcf()
fig.set_size_inches(50, 50)
sm = plt.cm.ScalarMappable(cmap=cmapC, norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = []
cbar=plt.colorbar(sm,aspect=40)
cbar.ax.tick_params(labelsize=30)

plt.figure(dpi=1200)
plt.show()
```


![png](https://raw.githubusercontent.com/samirak93/analytics/gh-pages/blog_images/images/blog3/output_5_0.png)


#### Statistics from the graph:
**Degree of the Nodes:**<br/> A histogram look at the degree distribution of the nodes. Degree of a node determines the no of nodes the parent node is connected directly. We can see that the large number of nodes have lower degree (<10) while only few nodes have degree >10.


```python
## Degree Histogram for sub-component
import collections
from bokeh.io import show, output_file
from bokeh.plotting import figure
import seaborn as sns
degree_sequence = sorted([d for n, d in g.subgraph(subnet).degree()], reverse=True)
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())

fig, ax = plt.subplots()
plt.bar(deg, cnt, width=0.8, color='r')
plt.title("Degree Histogram")
plt.ylabel("Count")
plt.xlabel("Degree")
fig = plt.gcf()
fig.set_size_inches(12,8)
plt.show()
```


![png](https://raw.githubusercontent.com/samirak93/analytics/gh-pages/blog_images/images/blog3/output_7_0.png)



```python
#Density of sub-component
density_sc=nx.density(g.subgraph(subnet))
density_sc
#Low value of density means less dense the nodes are and little collectiveness between the nodes. 
```




    0.0024622938288301533



#### Degree Centrality of the Nodes:
Degree Centrality of a nodes gives us the fraction of nodes each node it is connected to. Since 2 nodes have highest degree (53), the degree centrality of these nodes is the highest, followed by other nodes with higher degree. We can also see that there are nodes which have similar degree centrality.


```python
#Centrality of the Nodes.
degree_central=nx.degree_centrality(g.subgraph(subnet))
plt.bar(range(len(degree_central)), list(degree_central.values()),width=0.8, color='r')
# plt.xticks(range(len(degree_central)), list(degree_central.keys()))
fig = plt.gcf()
axes=plt.gca()
fig.set_size_inches(15,10)
axes.set_ylim([0,0.06])
plt.title("Centrality of Each node")
plt.show()
#degree centrality=fraction of nodes it is connected to. 2 nodes have highest centrality
```


![png](https://raw.githubusercontent.com/samirak93/analytics/gh-pages/blog_images/images/blog3/output_10_0.png)


#### Closeness Centrality:
Closeness centrality is defined by is the average length of the shortest path between the node and all other nodes in the graph. Thus the more central a node is, the closer it is to all other nodes. For this particular subgraph, the closeness centrality looks similar for most of the nodes.


```python
#Closeness Centrality of the Nodes.
close_degree_central=nx.closeness_centrality(g.subgraph(subnet))
plt.bar(range(len(close_degree_central)), list(close_degree_central.values()),width=0.8, color='r')
# plt.xticks(range(len(degree_central)), list(degree_central.keys()))
fig = plt.gcf()
axes=plt.gca()
fig.set_size_inches(15,10)
plt.title("Closeness Centrality of each node")
plt.show()
#Closeness Centrality-Closeness centrality of a node u is the reciprocal of the
#average shortest path distance to u over all n-1 reachable nodes.
```


![png](https://raw.githubusercontent.com/samirak93/analytics/gh-pages/blog_images/images/blog3/output_12_0.png)


#### Betweenness Centrality:
Betweenness centrality aims to find the vertex of the given graph. Betweenness centrality quantifies the number of times a node acts as a bridge along the shortest path between two other nodes. Betweenness centrality of a node v is the sum of the fraction of all-pairs shortest paths that pass- through v. In the below graph, we can see that few nodes act as the bridge for other nodes to be connected.


```python
#Between Centrality of the Nodes.
between_degree_central=nx.betweenness_centrality(g.subgraph(subnet))
plt.bar(range(len(between_degree_central)), list(between_degree_central.values()),width=0.8, color='r')
fig = plt.gcf()
axes=plt.gca()
fig.set_size_inches(15,10)
plt.title("Betweeness Centrality of each node")
plt.show()
#Between Centrality-Compute the shortest-path betweenness centrality for nodes.
#Betweenness centrality of a node v is the sum of the fraction of all-pairs shortest paths that pass through v
```


![png](https://raw.githubusercontent.com/samirak93/analytics/gh-pages/blog_images/images/blog3/output_14_0.png)


#### Eigen Vector Centrality:
Eigenvector centrality computes the centrality for a node based on the centrality of its neighbors. It is a relative scores to all nodes in the network based on the concept that connections to high-scoring nodes contribute more to the score of the node in question than equal connections to low-scoring nodes. In the subgraph, 2 nodes have high eigen vector score. Many other nodes have same score of 0.1, indicating that those nodes have similar in value.


```python
#Eigen Value Centrality of the Nodes.
Eigen_central=nx.eigenvector_centrality_numpy(g.subgraph(subnet))
plt.bar(range(len(Eigen_central)), list(Eigen_central.values()),width=0.8, color='r')
fig = plt.gcf()
axes=plt.gca()
fig.set_size_inches(15,10)
plt.title("Eigen Vector Centrality of each node")
plt.show()
#Eigenvector centrality computes the centrality for a node based on the centrality of its neighbors. 
#The eigenvector centrality for node i is the i-th element of the vector xdefined by the equation
```


![png](https://raw.githubusercontent.com/samirak93/analytics/gh-pages/blog_images/images/blog3/output_16_0.png)


#### Hub and Authority Score:
Hub score estimates the nodes value based on outgoing links. The authority score indicates the node value based on incoming links. The hub score and Authority score of the subgraph is same for all nodes, indicating that the no of incoming and outgoing nodes are same.


```python
#Hub Score of the Nodes.
hits_score=nx.hits_numpy(g.subgraph(subnet))
plt.bar(range(len(hits_score[0])), list(hits_score[0].values()),width=0.8, color='r')
fig = plt.gcf()
axes=plt.gca()
fig.set_size_inches(15,10)
plt.title("Hub Score for each node")
plt.show()
#Hub estimates the node value based on the Outgoing links.
```

    /Users/samirakumar/anaconda/lib/python2.7/site-packages/networkx/algorithms/link_analysis/hits_alg.py:207: ComplexWarning: Casting complex values to real discards the imaginary part
      hubs = dict(zip(G, map(float, h)))
    /Users/samirakumar/anaconda/lib/python2.7/site-packages/networkx/algorithms/link_analysis/hits_alg.py:208: ComplexWarning: Casting complex values to real discards the imaginary part
      authorities = dict(zip(G, map(float, a)))



![png](https://raw.githubusercontent.com/samirak93/analytics/gh-pages/blog_images/images/blog3/output_18_1.png)



```python
#Authority Score of the Nodes.
hits_score=nx.hits_numpy(g.subgraph(subnet))
plt.bar(range(len(hits_score[1])), list(hits_score[1].values()),width=0.8, color='r')
fig = plt.gcf()
axes=plt.gca()
fig.set_size_inches(15,10)
plt.title("Authority Score for each node")
plt.show()
#Authorities estimates the node value based on the incoming links.
```


![png](https://raw.githubusercontent.com/samirak93/analytics/gh-pages/blog_images/images/blog3/output_19_0.png)


#### Average Degree Neighbor:
Average degree of neighbor indicates the average degree of the neighbor for a given node. This gives us a good indication of the degree of the neighboring nodes for a given node. In the below graph, we can see that nodes only connected to 33 or 4429 directly have average degree 53 but the dispersion of the average degree is high.


```python
#Average Degree Neighbour
degree_assort=nx.average_neighbor_degree(g.subgraph(subnet))
plt.bar(range(len(degree_assort)), list(degree_assort.values()),width=0.8, color='r')
fig = plt.gcf()
axes=plt.gca()
fig.set_size_inches(15,10)
plt.title("Average degree neighbour for each node")
plt.show()
#Returns the average degree of the neighborhood of each node.
```


![png](https://raw.githubusercontent.com/samirak93/analytics/gh-pages/blog_images/images/blog3/output_21_0.png)


#### Diameter of Network:
The shortest distance/path length between the two most distant connected nodes in the network (the
longest of all the calculated path lengths). In the subgraph, the diameter is 41.



```python
#Diameter of the network graph
nx.diameter(g.subgraph(subnet))
#The shortest distance/path length between the two most distant connected nodes in the 
#network ( = the longest of all the calculated path lengths).
```




    41



#### Average Rating, Salesrank and Review Count for the nodes:
We calculated overall average for Rating, Salesrank and Review Count for the subgraph.



```python
filter_4429=df_final_group[(df_final_group['Source']==4429)|(df_final_group['Target']==4429)]
merged=pd.merge(filter_4429,df_filter,how='left', left_on=['Source'],right_on=['id'])
merged.loc[merged.id==4429,'title']='The Narcissistic Family : Diagnosis and Treatment'
merged.loc[merged.id==4429,'salesrank']=9727
merged.loc[merged.id==4429,'review_cnt']=19
merged.loc[merged.id==4429,'downloads']=19
merged.loc[merged.id==4429,'rating']=5.0
merged.loc[merged.id==4429,'id']=2501
merged[['rating','salesrank','review_cnt']].mean()
```




    rating            3.632075
    salesrank     75080.471698
    review_cnt       22.037736
    dtype: float64



## Digging Deeper:
Since the neighbor (source/target) for each product node can have an influence on the product purchase, we're finding the salesrank, rating and review count for each product.<br/> Example: If product 33 is associated with 55 and 66, then both 55 and 66 influence the purchase of 33. So we find the total salesrank, total reviews and rating for each product associated with each product. So for products 33 and 4429, we know they've highest degree (53). So for them we'd find salesrank, rating and review for the 53 products associated with them. This is done for all products in source and target.<br/>
<br/>
    The programming logic is simple, for source, we consider target and get salesrank of all the source products. We the consider source and get salesrank for all target products. We then groupby source and sum up all target salesrank. Then groupby target and sum up source targets. So for each product, we ge the total salesrank of all the products associated with it as either source or target.<br/> <br/>**Consider below example:**<br/><br/>
    
| source| target| salesrank_source  | salesrank_target|
| --- |---| ---| ---|
| 33| 55| 5|10|
| 33| 66|5|15|
|44| 33|20|5|

| id| Source_sum| Target_sum|
| --- |---| ---|
|33|25|20|
|44|5|0|
|55|5|0|
|66|5|0|

33 is source for 2 products. So their salesrank sum is 10+15=25 and 33 is target for one node. So its total salesrank is 20. So total salesrank for all products associated with 33 is 45 (10+15+20).
<br/> To find salesrank sum of neighbors of product 33, we first get salesrank of products 55 and 66 and store them as salesrank_target since they are in target column. Then we get salesrank of 44 and store them as salesrank_source. So now when we groupby source and sum salesrank_target, we store it as Source_sum. Then we groupby target and store the salesrank_source sum as Target_sum. We then add both Source_sum and Target_sum to get total salesrank for associated products for each product. Here's the catch. For some products, it can be present in only source/target. So if they're missing in either source/target, we consider them as 0 and make the combined dataframe.<br/><br/>Same process is followed for rating and review count.


```python
#Get the edges for each nodes from the network graph and store them in edges dataframe. From the initial table,
# we get the salesrank, rating and review detail for all the products. 
h=g.subgraph(subnet)
edges=pd.DataFrame(list(h.edges()))
edges=edges.rename(columns={'0':'Source','1':'Target'})

#total salesrank
df_cop_sales=pd.merge(edges,df_filter[['id','salesrank']],left_on=1,right_on='id',how='left')
df_cop_sales=df_cop_sales.rename(columns={'salesrank':'salesrank_target'})

df_cop_sale=pd.merge(edges,df_filter[['id','salesrank']],left_on=0,right_on='id',how='left')
df_cop_sale=df_cop_sale.rename(columns={'salesrank':'salesrank_source'})

df_cop_sale['salesrank_target']=df_cop_sales['salesrank_target']
df_source_sum=df_cop_sale.groupby(0)['salesrank_target'].sum().reset_index(name='Source_sum')
df_target_sum=df_cop_sale.groupby(1)['salesrank_source'].sum().reset_index(name='Target_sum')

x = df_source_sum.set_index(0)
y = df_target_sum.set_index(1).rename_axis(0)
y.columns = x.columns

combined=y.add(x, fill_value=0)
combined=pd.DataFrame(combined)
combined=combined.rename(columns={'0':'id','Source_sum':'Total_salesrank'})
combined.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Total_salesrank</th>
    </tr>
    <tr>
      <th>0</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>33</th>
      <td>4354123.0</td>
    </tr>
    <tr>
      <th>77</th>
      <td>125232.0</td>
    </tr>
    <tr>
      <th>78</th>
      <td>804969.0</td>
    </tr>
    <tr>
      <th>130</th>
      <td>159895.0</td>
    </tr>
    <tr>
      <th>148</th>
      <td>60293.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Total rating
df_cop_rating=pd.merge(edges,df_filter[['id','rating']],left_on=1,right_on='id',how='left')
df_cop_rating=df_cop_rating.rename(columns={'rating':'rating_target'})

df_cop_rate=pd.merge(edges,df_filter[['id','rating']],left_on=0,right_on='id',how='left')
df_cop_rate=df_cop_rate.rename(columns={'rating':'rating_source'})
df_cop_rate['rating_target']=df_cop_rating['rating_target']

df_source_rate_sum=df_cop_rate.groupby(0)['rating_target'].sum().reset_index(name='Source_sum')
df_target_rate_sum=df_cop_rate.groupby(1)['rating_source'].sum().reset_index(name='Target_sum')

x = df_source_rate_sum.set_index(0)
y = df_target_rate_sum.set_index(1).rename_axis(0)
y.columns = x.columns

combined_rating=y.add(x, fill_value=0)
combined_rating=pd.DataFrame(combined_rating)
combined_rating=combined_rating.rename(columns={'0':'id','Source_sum':'Total_rating'})
combined_rating.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Total_rating</th>
    </tr>
    <tr>
      <th>0</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>33</th>
      <td>217.5</td>
    </tr>
    <tr>
      <th>77</th>
      <td>14.0</td>
    </tr>
    <tr>
      <th>78</th>
      <td>49.5</td>
    </tr>
    <tr>
      <th>130</th>
      <td>9.0</td>
    </tr>
    <tr>
      <th>148</th>
      <td>4.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Total review count
df_cop_reviews=pd.merge(edges,df_filter[['id','review_cnt']],left_on=1,right_on='id',how='left')
df_cop_reviews=df_cop_reviews.rename(columns={'review_cnt':'review_cnt_target'})

df_cop_review=pd.merge(edges,df_filter[['id','review_cnt']],left_on=0,right_on='id',how='left')
df_cop_review=df_cop_review.rename(columns={'review_cnt':'review_cnt_source'})
df_cop_review['review_cnt_target']=df_cop_reviews['review_cnt_target']

df_source_review_sum=df_cop_review.groupby(0)['review_cnt_target'].sum().reset_index(name='Source_sum')
df_source_reviews_sum=df_cop_review.groupby(1)['review_cnt_source'].sum().reset_index(name='Target_sum')

x = df_source_review_sum.set_index(0)
y = df_source_reviews_sum.set_index(1).rename_axis(0)
y.columns = x.columns

combined_reviews=y.add(x, fill_value=0)
combined_reviews=pd.DataFrame(combined_reviews)
combined_reviews=combined_reviews.rename(columns={'0':'id','Source_sum':'Total_review_cnt'})
combined_reviews.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Total_review_cnt</th>
    </tr>
    <tr>
      <th>0</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>33</th>
      <td>1117.0</td>
    </tr>
    <tr>
      <th>77</th>
      <td>12.0</td>
    </tr>
    <tr>
      <th>78</th>
      <td>1736.0</td>
    </tr>
    <tr>
      <th>130</th>
      <td>9.0</td>
    </tr>
    <tr>
      <th>148</th>
      <td>14.0</td>
    </tr>
  </tbody>
</table>
</div>



#### Average salesrank, rating and review count: 
By finding out the degree for each node, we can find the average salesrank, rating and review count for each product.


```python
degrees=pd.DataFrame(list(h.degree()))
degrees=degrees.sort_values(0)
degrees.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14</th>
      <td>33</td>
      <td>53</td>
    </tr>
    <tr>
      <th>211</th>
      <td>77</td>
      <td>3</td>
    </tr>
    <tr>
      <th>140</th>
      <td>78</td>
      <td>11</td>
    </tr>
    <tr>
      <th>57</th>
      <td>130</td>
      <td>2</td>
    </tr>
    <tr>
      <th>776</th>
      <td>148</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_merge_sales=pd.merge(combined,degrees,left_on=0,right_on=0,how='left')
df_merge_sales['Average_Salesrank']=df_merge_sales['Total_salesrank']/df_merge_sales[1]
df_merge_sales=df_merge_sales.rename(columns={'key_0':'id',1:'Degree'})
df_merge_sales.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>Total_salesrank</th>
      <th>Degree</th>
      <th>Average_Salesrank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>33</td>
      <td>4354123.0</td>
      <td>53</td>
      <td>82153.264151</td>
    </tr>
    <tr>
      <th>1</th>
      <td>77</td>
      <td>125232.0</td>
      <td>3</td>
      <td>41744.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>78</td>
      <td>804969.0</td>
      <td>11</td>
      <td>73179.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>130</td>
      <td>159895.0</td>
      <td>2</td>
      <td>79947.500000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>148</td>
      <td>60293.0</td>
      <td>2</td>
      <td>30146.500000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_merge_rating=pd.merge(combined_rating,degrees,left_on=0,right_on=0,how='left')
df_merge_rating['Average_Rating']=df_merge_rating['Total_rating']/df_merge_rating[1]
df_merge_rating=df_merge_rating.rename(columns={'key_0':'id',1:'Degree'})
df_merge_rating.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>Total_rating</th>
      <th>Degree</th>
      <th>Average_Rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>33</td>
      <td>217.5</td>
      <td>53</td>
      <td>4.103774</td>
    </tr>
    <tr>
      <th>1</th>
      <td>77</td>
      <td>14.0</td>
      <td>3</td>
      <td>4.666667</td>
    </tr>
    <tr>
      <th>2</th>
      <td>78</td>
      <td>49.5</td>
      <td>11</td>
      <td>4.500000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>130</td>
      <td>9.0</td>
      <td>2</td>
      <td>4.500000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>148</td>
      <td>4.5</td>
      <td>2</td>
      <td>2.250000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_merge_reviews=pd.merge(combined_reviews,degrees,left_on=0,right_on=0,how='left')
df_merge_reviews['Average_Reviews']=df_merge_reviews['Total_review_cnt']/df_merge_reviews[1]
df_merge_reviews=df_merge_reviews.rename(columns={'key_0':'id',1:'Degree'})
df_merge_reviews.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>Total_review_cnt</th>
      <th>Degree</th>
      <th>Average_Reviews</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>33</td>
      <td>1117.0</td>
      <td>53</td>
      <td>21.075472</td>
    </tr>
    <tr>
      <th>1</th>
      <td>77</td>
      <td>12.0</td>
      <td>3</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>78</td>
      <td>1736.0</td>
      <td>11</td>
      <td>157.818182</td>
    </tr>
    <tr>
      <th>3</th>
      <td>130</td>
      <td>9.0</td>
      <td>2</td>
      <td>4.500000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>148</td>
      <td>14.0</td>
      <td>2</td>
      <td>7.000000</td>
    </tr>
  </tbody>
</table>
</div>



#### Network Graph parameters:
From the network graph, we're finding each parameters for each node and adding them to the dataframe.


```python
centrality_df=pd.DataFrame(degree_central.items(),columns=['id','centrality'])
degree_df=pd.DataFrame(degrees.items(),columns=['id','degree'])
closeness_central_df=pd.DataFrame(close_degree_central.items(),columns=['id','closeness_centrality'])
between_degree_df=pd.DataFrame(between_degree_central.items(),columns=['id','between_centrality'])
Eigen_central_df=pd.DataFrame(Eigen_central.items(),columns=['id','eigen_centrality'])
hubs_score_df=pd.DataFrame(hits_score[0].items(),columns=['id','hub_score'])
authority_score_df=pd.DataFrame(hits_score[1].items(),columns=['id','authority_score'])
avg_degree_neighbour_df=pd.DataFrame(degree_assort.items(),columns=['id','avg_degree_neighbour'])
centrality_df.head()
#filter purchase dataset
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>centrality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>24578</td>
      <td>0.002215</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4099</td>
      <td>0.004430</td>
    </tr>
    <tr>
      <th>2</th>
      <td>141316</td>
      <td>0.002215</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2057</td>
      <td>0.003322</td>
    </tr>
    <tr>
      <th>4</th>
      <td>122893</td>
      <td>0.002215</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_filter_books=df_filter[(df_filter.id.isin(centrality_df.id)) & (df_filter.id.isin(degree_df.id))
                          &(df_filter.id.isin(between_degree_df.id))&(df_filter.id.isin(Eigen_central_df.id))
                         &(df_filter.id.isin(hubs_score_df.id))&(df_filter.id.isin(authority_score_df.id))
                         &(df_filter.id.isin(avg_degree_neighbour_df.id))]
df_filter_books.head()
df_filter_books=pd.merge(df_filter_books,centrality_df,left_on='id',right_on='id',how='left')
df_filter_books=pd.merge(df_filter_books,degree_df,left_on='id',right_on='id',how='left')
df_filter_books=pd.merge(df_filter_books,closeness_central_df,left_on='id',right_on='id',how='left')
df_filter_books=pd.merge(df_filter_books,between_degree_df,left_on='id',right_on='id',how='left')
df_filter_books=pd.merge(df_filter_books,Eigen_central_df,left_on='id',right_on='id',how='left')
df_filter_books=pd.merge(df_filter_books,hubs_score_df,left_on='id',right_on='id',how='left')
df_filter_books=pd.merge(df_filter_books,authority_score_df,left_on='id',right_on='id',how='left')
df_filter_books=pd.merge(df_filter_books,avg_degree_neighbour_df,left_on='id',right_on='id',how='left')

df_filter_books=pd.merge(df_filter_books,df_merge_sales[['key_0','Average_Salesrank']],left_on='id',right_on='key_0',how='left')
df_filter_books=pd.merge(df_filter_books,df_merge_rating[['key_0','Average_Rating']],left_on='id',right_on='key_0',how='left')
df_filter_books=pd.merge(df_filter_books,df_merge_reviews[['key_0','Average_Reviews']],left_on='id',right_on='key_0',how='left')
df_filter_books=df_filter_books.drop('key_0_x',1)
df_filter_books=df_filter_books.drop('key_0_y',1)
df_filter_books=df_filter_books.drop('key_0',1)
df_filter_books.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>title</th>
      <th>group</th>
      <th>salesrank</th>
      <th>review_cnt</th>
      <th>downloads</th>
      <th>rating</th>
      <th>centrality</th>
      <th>degree</th>
      <th>closeness_centrality</th>
      <th>between_centrality</th>
      <th>eigen_centrality</th>
      <th>hub_score</th>
      <th>authority_score</th>
      <th>avg_degree_neighbour</th>
      <th>Average_Salesrank</th>
      <th>Average_Rating</th>
      <th>Average_Reviews</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>33</td>
      <td>Double Jeopardy (T*Witches, 6)</td>
      <td>Book</td>
      <td>97166.0</td>
      <td>4</td>
      <td>4</td>
      <td>5.0</td>
      <td>0.058693</td>
      <td>53</td>
      <td>0.145598</td>
      <td>0.586724</td>
      <td>4.542173e-02</td>
      <td>6.067368e-03</td>
      <td>6.067368e-03</td>
      <td>2.132075</td>
      <td>82153.264151</td>
      <td>4.103774</td>
      <td>21.075472</td>
    </tr>
    <tr>
      <th>1</th>
      <td>77</td>
      <td>Water Touching Stone</td>
      <td>Book</td>
      <td>27012.0</td>
      <td>11</td>
      <td>11</td>
      <td>4.5</td>
      <td>0.003322</td>
      <td>3</td>
      <td>0.081682</td>
      <td>0.013235</td>
      <td>7.515199e-08</td>
      <td>1.003869e-08</td>
      <td>1.003869e-08</td>
      <td>2.000000</td>
      <td>41744.000000</td>
      <td>4.666667</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>78</td>
      <td>The Ebony Cookbook: A Date With a Dish</td>
      <td>Book</td>
      <td>140480.0</td>
      <td>3</td>
      <td>3</td>
      <td>4.5</td>
      <td>0.012182</td>
      <td>11</td>
      <td>0.107615</td>
      <td>0.053657</td>
      <td>3.038877e-05</td>
      <td>4.059287e-06</td>
      <td>4.059287e-06</td>
      <td>2.909091</td>
      <td>73179.000000</td>
      <td>4.500000</td>
      <td>157.818182</td>
    </tr>
    <tr>
      <th>3</th>
      <td>130</td>
      <td>The O'Reilly Factor: The Good, the Bad, and th...</td>
      <td>Book</td>
      <td>29460.0</td>
      <td>375</td>
      <td>375</td>
      <td>3.5</td>
      <td>0.002215</td>
      <td>2</td>
      <td>0.097338</td>
      <td>0.017581</td>
      <td>4.084168e-06</td>
      <td>5.455571e-07</td>
      <td>5.455571e-07</td>
      <td>6.500000</td>
      <td>79947.500000</td>
      <td>4.500000</td>
      <td>4.500000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>148</td>
      <td>Firebird</td>
      <td>Book</td>
      <td>77008.0</td>
      <td>42</td>
      <td>42</td>
      <td>4.0</td>
      <td>0.002215</td>
      <td>2</td>
      <td>0.091194</td>
      <td>0.004425</td>
      <td>7.562043e-07</td>
      <td>1.010127e-07</td>
      <td>1.010127e-07</td>
      <td>12.000000</td>
      <td>30146.500000</td>
      <td>2.250000</td>
      <td>7.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_log=df_filter_books

df_log['review_cnt']=np.log(df_log['review_cnt']+1)
df_log['downloads']=np.log(df_log['downloads']+1)
df_log['rating']=np.log(df_log['rating']+1)
df_log['centrality']=np.log(df_log['centrality']+1)
df_log['degree']=np.log(df_log['degree']+1)
df_log['closeness_centrality']=np.log(df_log['closeness_centrality']+1)
df_log['between_centrality']=np.log(df_log['between_centrality']+1)
df_log['eigen_centrality']=np.log(df_log['eigen_centrality']+1)
df_log['hub_score']=np.log(df_log['hub_score']+1)
df_log['authority_score']=np.log(df_log['authority_score']+1)
df_log['avg_degree_neighbour']=np.log(df_log['avg_degree_neighbour']+1)
df_log['Average_Salesrank']=np.log(df_log['Average_Salesrank']+1)
df_log['Average_Rating']=np.log(df_log['Average_Rating']+1)
df_log['Average_Reviews']=np.log(df_log['Average_Reviews']+1)
# df_dummy['nghb_mn_rating_y']=np.log(df_dummy['nghb_mn_rating_y']+1)
# df_dummy['nghb_mn_review']=np.log(df_dummy['nghb_mn_review']+1)


df_log.tail()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>title</th>
      <th>group</th>
      <th>salesrank</th>
      <th>review_cnt</th>
      <th>downloads</th>
      <th>rating</th>
      <th>centrality</th>
      <th>degree</th>
      <th>closeness_centrality</th>
      <th>between_centrality</th>
      <th>eigen_centrality</th>
      <th>hub_score</th>
      <th>authority_score</th>
      <th>avg_degree_neighbour</th>
      <th>Average_Salesrank</th>
      <th>Average_Rating</th>
      <th>Average_Reviews</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>899</th>
      <td>261523</td>
      <td>Ghosts of Tsavo: Tracking the Mythic Lions of ...</td>
      <td>Book</td>
      <td>91750.0</td>
      <td>2.484907</td>
      <td>2.484907</td>
      <td>1.609438</td>
      <td>0.002212</td>
      <td>1.098612</td>
      <td>0.094531</td>
      <td>0.004415</td>
      <td>0.001640</td>
      <td>0.000219</td>
      <td>0.000219</td>
      <td>1.098612</td>
      <td>10.751596</td>
      <td>1.658228</td>
      <td>1.504077</td>
    </tr>
    <tr>
      <th>900</th>
      <td>261524</td>
      <td>Object-Oriented Programming in Common Lisp: A ...</td>
      <td>Book</td>
      <td>79520.0</td>
      <td>1.386294</td>
      <td>1.386294</td>
      <td>1.704748</td>
      <td>0.002212</td>
      <td>1.098612</td>
      <td>0.086387</td>
      <td>0.002212</td>
      <td>0.000221</td>
      <td>0.000029</td>
      <td>0.000029</td>
      <td>0.916291</td>
      <td>10.933089</td>
      <td>1.609438</td>
      <td>5.049856</td>
    </tr>
    <tr>
      <th>901</th>
      <td>261898</td>
      <td>How To Be A Para Pro : A Comprehensive Trainin...</td>
      <td>Book</td>
      <td>122234.0</td>
      <td>1.945910</td>
      <td>1.945910</td>
      <td>1.704748</td>
      <td>0.002212</td>
      <td>1.098612</td>
      <td>0.106947</td>
      <td>0.002212</td>
      <td>0.000820</td>
      <td>0.000110</td>
      <td>0.000110</td>
      <td>0.916291</td>
      <td>11.861781</td>
      <td>1.658228</td>
      <td>2.079442</td>
    </tr>
    <tr>
      <th>902</th>
      <td>261899</td>
      <td>The Listening Walk</td>
      <td>Book</td>
      <td>146686.0</td>
      <td>1.386294</td>
      <td>1.386294</td>
      <td>1.704748</td>
      <td>0.002212</td>
      <td>1.098612</td>
      <td>0.119722</td>
      <td>0.004415</td>
      <td>0.006086</td>
      <td>0.000815</td>
      <td>0.000815</td>
      <td>3.349904</td>
      <td>11.605514</td>
      <td>1.749200</td>
      <td>1.791759</td>
    </tr>
    <tr>
      <th>903</th>
      <td>261966</td>
      <td>The Best Little Beading Book (Beadwork Books)</td>
      <td>Book</td>
      <td>136801.0</td>
      <td>2.484907</td>
      <td>2.484907</td>
      <td>1.609438</td>
      <td>0.001107</td>
      <td>0.693147</td>
      <td>0.096616</td>
      <td>0.000000</td>
      <td>0.000108</td>
      <td>0.000014</td>
      <td>0.000014</td>
      <td>1.098612</td>
      <td>11.713701</td>
      <td>1.704748</td>
      <td>1.945910</td>
    </tr>
  </tbody>
</table>
</div>


#### Regression model to predict the salesrank of the product given other parameters.


```python
import statsmodels.api as sm
import statsmodels.formula.api as smf

from statsmodels.genmod.families import Poisson,Binomial

X = df_log[['review_cnt','downloads','rating','centrality','between_centrality','closeness_centrality','eigen_centrality','authority_score'
              ,'avg_degree_neighbour','Average_Salesrank','Average_Rating','Average_Reviews']]
X = sm.add_constant(X)

y = df_log.salesrank

poisson_model = smf.poisson('salesrank ~ review_cnt+downloads+ rating+centrality+between_centrality+closeness_centrality+eigen_centrality+authority_score + avg_degree_neighbour+ Average_Salesrank+Average_Rating+Average_Reviews', df_log)

res=poisson_model.fit(method='bfgs')

res.summary()
```

    Warning: Desired error not necessarily achieved due to precision loss.
             Current function value: 14142.163464
             Iterations: 34
             Function evaluations: 43
             Gradient evaluations: 42


    /Users/samirakumar/anaconda/lib/python2.7/site-packages/statsmodels/base/model.py:508: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
      "Check mle_retvals", ConvergenceWarning)





<table class="simpletable">
<caption>Poisson Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>     <td>salesrank</td>    <th>  No. Observations:  </th>   <td>   904</td>   
</tr>
<tr>
  <th>Model:</th>              <td>Poisson</td>     <th>  Df Residuals:      </th>   <td>   891</td>   
</tr>
<tr>
  <th>Method:</th>               <td>MLE</td>       <th>  Df Model:          </th>   <td>    12</td>   
</tr>
<tr>
  <th>Date:</th>          <td>Wed, 07 Nov 2018</td> <th>  Pseudo R-squ.:     </th>   <td>0.09718</td>  
</tr>
<tr>
  <th>Time:</th>              <td>19:08:52</td>     <th>  Log-Likelihood:    </th> <td>-1.2785e+07</td>
</tr>
<tr>
  <th>converged:</th>           <td>False</td>      <th>  LL-Null:           </th> <td>-1.4161e+07</td>
</tr>
<tr>
  <th> </th>                      <td> </td>        <th>  LLR p-value:       </th>   <td> 0.000</td>   
</tr>
</table>
<table class="simpletable">
<tr>
            <td></td>              <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>            <td>   10.9255</td> <td>    0.002</td> <td> 5185.527</td> <td> 0.000</td> <td>   10.921</td> <td>   10.930</td>
</tr>
<tr>
  <th>review_cnt</th>           <td>   -0.3315</td> <td>    0.003</td> <td> -129.287</td> <td> 0.000</td> <td>   -0.337</td> <td>   -0.326</td>
</tr>
<tr>
  <th>downloads</th>            <td>    0.1655</td> <td>    0.003</td> <td>   64.490</td> <td> 0.000</td> <td>    0.160</td> <td>    0.171</td>
</tr>
<tr>
  <th>rating</th>               <td>    0.1016</td> <td>    0.000</td> <td>  390.267</td> <td> 0.000</td> <td>    0.101</td> <td>    0.102</td>
</tr>
<tr>
  <th>centrality</th>           <td>   -8.1959</td> <td>    0.067</td> <td> -122.140</td> <td> 0.000</td> <td>   -8.327</td> <td>   -8.064</td>
</tr>
<tr>
  <th>between_centrality</th>   <td>    0.4435</td> <td>    0.005</td> <td>   89.255</td> <td> 0.000</td> <td>    0.434</td> <td>    0.453</td>
</tr>
<tr>
  <th>closeness_centrality</th> <td>   -0.6735</td> <td>    0.008</td> <td>  -85.707</td> <td> 0.000</td> <td>   -0.689</td> <td>   -0.658</td>
</tr>
<tr>
  <th>eigen_centrality</th>     <td>  -16.5839</td> <td>    0.052</td> <td> -316.531</td> <td> 0.000</td> <td>  -16.687</td> <td>  -16.481</td>
</tr>
<tr>
  <th>authority_score</th>      <td>  109.3204</td> <td>    0.342</td> <td>  319.884</td> <td> 0.000</td> <td>  108.651</td> <td>  109.990</td>
</tr>
<tr>
  <th>avg_degree_neighbour</th> <td>    0.0753</td> <td>    0.000</td> <td>  380.543</td> <td> 0.000</td> <td>    0.075</td> <td>    0.076</td>
</tr>
<tr>
  <th>Average_Salesrank</th>    <td>    0.0430</td> <td>    0.000</td> <td>  241.128</td> <td> 0.000</td> <td>    0.043</td> <td>    0.043</td>
</tr>
<tr>
  <th>Average_Rating</th>       <td>   -0.0593</td> <td>    0.000</td> <td> -161.523</td> <td> 0.000</td> <td>   -0.060</td> <td>   -0.059</td>
</tr>
<tr>
  <th>Average_Reviews</th>      <td>    0.0038</td> <td>    0.000</td> <td>   32.092</td> <td> 0.000</td> <td>    0.004</td> <td>    0.004</td>
</tr>
</table>




```python
y_pred = res.predict(X)
from sklearn.metrics import mean_squared_error,mean_absolute_error
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
print mse,mae
```

    1799617311.4698195 36367.79088431795



```python
print('Parameters: ', res.params)
print('T-values: ', res.tvalues)
from sklearn.metrics import accuracy_score,r2_score

errors = abs(y_pred - y)

print('Variance score: %.2f' % r2_score(y, y_pred))
print('Mean Absolute Error:', round(np.mean(errors), 2), 'salesrank.')

mape = 100 * (errors / y)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
```

    ('Parameters: ', Intercept                10.925529
    review_cnt               -0.331510
    downloads                 0.165527
    rating                    0.101626
    centrality               -8.195922
    between_centrality        0.443548
    closeness_centrality     -0.673488
    eigen_centrality        -16.583895
    authority_score         109.320403
    avg_degree_neighbour      0.075330
    Average_Salesrank         0.043040
    Average_Rating           -0.059289
    Average_Reviews           0.003754
    dtype: float64)
    ('T-values: ', Intercept               5185.527173
    review_cnt              -129.287415
    downloads                 64.489734
    rating                   390.267055
    centrality              -122.140239
    between_centrality        89.255351
    closeness_centrality     -85.706590
    eigen_centrality        -316.531474
    authority_score          319.883670
    avg_degree_neighbour     380.543409
    Average_Salesrank        241.127911
    Average_Rating          -161.522590
    Average_Reviews           32.092003
    dtype: float64)
    Variance score: 0.10
    ('Mean Absolute Error:', 36367.79, 'salesrank.')
    ('Accuracy:', -215.1, '%.')



```python
y_pred[0:5],y[0:5]
```




    (0    54117.163709
     1    63320.558265
     2    75090.616702
     3    38723.365700
     4    57808.631127
     dtype: float64, 0     97166.0
     1     27012.0
     2    140480.0
     3     29460.0
     4     77008.0
     Name: salesrank, dtype: float64)




```python
fig=plt.gcf()
fig.set_size_inches(10,8)
# plt.scatter(y,y_pred)
# plt.title('Train dataset Real vs. Predicted Values')
# plt.show()
import seaborn as sns
sns.regplot(y,y_pred)
plt.show()
```


![png](https://raw.githubusercontent.com/samirak93/analytics/gh-pages/blog_images/images/blog3/output_44_0.png)



```python
fig=plt.gcf()
fig.set_size_inches(10,8)


X = sm.add_constant(X)
model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
sns.regplot(df_log['salesrank'], model.resid_deviance, fit_reg=False)
plt.title('Residual plot')
plt.xlabel('Salesrank')
plt.ylabel('Residuals')
```




    Text(0,0.5,'Residuals')




![png](https://raw.githubusercontent.com/samirak93/analytics/gh-pages/blog_images/images/blog3/output_45_1.png)


### Interpretation from model:
The value of intercept has a meaning here since a product can be purchased alone and it can have a sales rank even if it doesn’t have any co-product.
P-values of all the variables are less than 0.05, indicating that they’re significant in the model. As seen in the R model, the variables used are significant in predicting the salesrank.
The model was initially built with large no of variables and the insignificant variables were later dropped from the model.
The authority score has the highest coefficient, indicating that for every value increase in authority score, the salesrank tends to increase. The authority score is no of incoming nodes. So products which are purchased as target product tends to have higher sales rank than products that are purchased as source product.
The final model’s mean absolute deviation value is about 210.55 and mean of residuals is 28885 indicating a good fit of the parameters.
