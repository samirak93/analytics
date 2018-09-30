---
layout: post
title: ' Dynamic Network Pass Map'
---

This blog post will look at building a dynamic networkx graph using **Bokeh** and **NetworkX** libraries in python.  
This is just a experimentation of the concept.

The data being used for this graph is the pass combination between football (soccer) players from a match. The data was scraped from few  online websites. 

I've already pre-processed the data to make the following columns.
***From, To**: Pass from and to player names.
 **Game_Time_Start, Game_Time_End**: Game time (in seconds) when the pass starts and ends, 
 **Start_x, Start_y**: Pass start x and y coordinates,
 **End_x, End_y**: Pass end x and y coordinates*. 
 

Bokeh provides a nice option to plot NetworkX graphs on the web browser. The problem was plotting the graph nodes in custom position (player location on the field) rather than fixed layout like spring circular or any other layout. 

With the help of few people on the Bokeh google user group, I was able to sort out that problem. 

Going into the [code](https://github.com/samirak93/dynamic-pass-network/blob/master/myapp/main.py), we import the necessary libraries and packages. 

    import pandas as pd  
    import numpy as np  
    from bokeh.models.widgets import RangeSlider,Div  
    from bokeh.models import HoverTool  
    from bokeh.io import curdoc  
    from bokeh.layouts import column,layout  
    import networkx as nx  
    from bokeh.models.graphs import from_networkx  
    from bokeh.models import StaticLayoutProvider,Circle,LabelSet,ColumnDataSource,CustomJS  
    from bokeh.plotting import figure

Then we're loading the csv file and reading the columns and observations.

    df = pd.read_csv('myapp/data/final_data.csv',encoding='utf-8')
    pass_player=pd.DataFrame(df)
    final_data=pass_player.groupby(['From','To','Game_Time_Start','Game_Time_End','Start_x','Start_y','End_x','End_y']).size().reset_index(name="Freq")

The "***Freq***" in the final_data will be now be ***1*** for all rows. This will later update as the slider value (time frame) are considered.  

## player_plot function

The player_plot() function plots the NetworkX using bokeh. 

    lower = np.round(range_slider.value[0]) #low range of slider value
    higher = np.round(range_slider.value[1]) #high range of slider value


    #filter the data based on slider values
    filter_data = final_data[(final_data['Game_Time_Start']>=lower )& (final_data['Game_Time_Start']<=higher)]
    #size of the edges as Freq
        size = filter_data.groupby(['From','To']).size().reset_index(name="Freq")
    #group the player locations based on where pass starts
        grouped = filter_data.groupby(['To'])[['Start_x','Start_y']].mean().reset_index()


The following code will build the graph using NetworkX

	    G = nx.DiGraph()

	    for index, row in grouped.iterrows():
            G.add_node(row['To'],pos=row[['Start_x','Start_y']])
    
        for index, row in size.iterrows():
            G.add_edge(row['From'], row['To'],weight=row['Freq'])
    
    
        fixed_pos=grouped.set_index('To').T.to_dict('list')
        fixed_nodes = fixed_pos.keys()
        pos=nx.get_node_attributes(G,'pos')
    
        edges = G.edges()
    
        weights = [G[u][v]['weight'] for u,v in edges]


The ***from_networkx*** function in bokeh converts the graph to default layout like spring, circular or any other. 

    graph = from_networkx(G,nx.spring_layout)  
    #Use the StaticLayoutProvider to convert the spring_layout back to user defined layout
    fixed_layout_provider = StaticLayoutProvider(graph_layout=pos)  
    graph.layout_provider = fixed_layout_provider
    graph.node_renderer.glyph = Circle(size=20,fill_color='orangered')  
      
    plot.xgrid.grid_line_color = None  
    plot.ygrid.grid_line_color = None  
    plot.axis.visible=False
    #use the line_width to indicate the weight of each node. Thicker line means larger freq between 2 nodes.
    graph.edge_renderer.data_source.data["line_width"] = [G.get_edge_data(a,b)['weight'] for a, b in G.edges()]
    graph.edge_renderer.glyph.line_width = {'field': 'line_width'}
    plot.renderers.append(graph)  
    pos_values=np.array(fixed_pos.values())  
      
    coordinates=pd.DataFrame(pos_values,columns=['x','y'])  
    coordinates['player'] =fixed_pos.keys()  
    source = ColumnDataSource(data=dict(x=coordinates.x,y=coordinates.y,player=coordinates.player))  
    labels = LabelSet(x='x', y='y', text='player', source=source,x_offset=-45, y_offset=-25,text_color='black',render_mode='canvas',text_font_size='10pt')  
    plot.renderers.append(labels)  
    return plot

The rest of the code are self explanatory. 
The entire code can be found [here](https://github.com/samirak93/dynamic-pass-network/blob/master/myapp/main.py).

I've hosted the final output on Heroku App. [App Link](https://dynamic-pass-network.herokuapp.com/myapp).

In case you find it difficult to host bokeh plots on heroku, please read my [other blog post](https://samirak93.github.io/analytics/Deploy-bokeh-server-plots-using-heroku.html) where I've explained the process.

Leave your feedback below or contact me using [Twitter](https://twitter.com/Samirak93) or [LinkedIN](http://linkedin.com/in/samirakumar/) or you can e-mail me (e-mail id on my contact page).

Thanks.
