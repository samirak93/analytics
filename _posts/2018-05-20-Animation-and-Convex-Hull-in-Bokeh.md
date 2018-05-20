
This blog post looks at creating an animation slider (with Play and Pause buttons) to plot 2D coordinates of player movement in a soccer game. Also, this post explains the steps to create a toggle button, to show/hide the convex hull plot of the teams. I've used [Bokeh](https://bokeh.pydata.org) to plot the viz. Bokeh gives a good looking viz in the browser and also provides smooth interface for animation. I've also tried the same with [Matplotlib](https://matplotlib.org) and it was successful. But the features of bokeh (compared to matplotlib) makes it a better choice. I could write another separate blog post, if you're interested to know the same process in Matplotlib. 

The dataset used in the post is from **STATS**. You can submit an [online](https://www.stats.com/data-science/) request to obtain the dataset from them. The data used in the viz is not shared.

Note that I've already done some pre-processing to clean the data since the raw data from STATS doesn't have player/team names/ID and we'd have to manually add them. I'm not going to explain in depth about the data cleaning process. Once you get the dataset and have doubts on adding team/player tags, do contact me on twitter/mail.

Once you've the data file ready, create the below folder directory, which would be needed to run the bokeh server.


myapp<br/>
   |<br/>
   +---data **(folder)**<br/>
   |    +---data.csv<br/>
   |<br/>
   +---main.py<br/>
   +---static **(folder)**<br/>
   |    +---images **(folder within static)**<br/>
   |    |    +---pitch_image.png<br/>
   

Once the player and team tags are added, we'd have the data as below

    import pandas as pd
    df=pd.read_csv('data/data.csv')
    headers = ["x", "y", "team_id", "player_id","time"]  
    all_team = pd.DataFrame(df, columns=headers)
    

 Here, 
 x and y: Player coordinates on the pitch, 
 team_id: 1 and 2 for both teams respectively, 3 for ball, 
 player_id: Ranges from 1-11 for both teams, 12 for ball
 time: time of sequence. 


Now import all necessary packages.

    import numpy as np  
    import pandas as pd  
    from bokeh.io import curdoc  
    from bokeh.layouts import row, widgetbox,column  
    from bokeh.models import ColumnDataSource,LabelSet,PointDrawTool,CustomJS  
    from bokeh.models.widgets import Slider,Paragraph,Button,CheckboxButtonGroup  
    from bokeh.plotting import figure  
    from scipy.spatial import ConvexHull

My game sequence time starts at 0. So I've created a variable to initialise time and calculated player coordinates based on that. Since my slider will always start at 0, I've initialised the time as 0.

    i=0  
      
    x = all_team[all_team.time==i].x  #Calculate x based on time
    y = all_team[all_team.time==i].y  #Calculate y based on time


To plot the player labels and team colour, I created 2 separate variables **player_id** and **c**. This is bit like hard coding the labels and colour, but since these will not change for the whole sequence, it doesn't matter if we hard code this or use any other method. You could also use **factor_cmap** for player color. 
 
 

    player_id=['1','2','3','4','5','6','7','8','9','10','11','1','2','3','4','5','6','7','8','9','10','11',' ']  
    
    #The last value is left empty to not show label for the ball.
    
    c=['dodgerblue','dodgerblue','dodgerblue','dodgerblue','dodgerblue','dodgerblue','dodgerblue','dodgerblue','dodgerblue','dodgerblue','dodgerblue','orangered','orangered','orangered','orangered','orangered','orangered','orangered','orangered','orangered','orangered','orangered','gold']


Once we've the required variables, create the **ColumnDataSource** as below.

    source = ColumnDataSource(data=dict(x=x, y=y,player_id=player_id,color=c))

Now we should plot the background image (football pitch). For this, store the image inside the static/images folder within myapp folder.

Now create the plot as below. 

    #Set up plot  
    plot = figure(name='base',plot_height=600, plot_width=800, title="Game Animation",  
                  tools="reset,save",  
                  x_range=[-52.5,52.5], y_range=[-34, 34],toolbar_location="below")
                  
    plot.image_url(url=["myapp/static/images/base.png"],x=-52.5,y=-34,w=105,h=68,anchor="bottom_left")


    #Plot the player coordinates
    st=plot.scatter('x','y', source=source,size=20,fill_color='color')

    #Add label to the scatter plot
    labels=LabelSet(x='x', y='y', text='player_id', level='glyph',  
                      x_offset=-5, y_offset=-7, source=source, render_mode='canvas',text_color='white',text_font_size="10pt")
                      
    plot.add_layout(labels)

    layout = column(row(plot))  #add plot to layout
      
    curdoc().add_root(layout)
    
    curdoc().title = "Game Animation"

Now run the bokeh server using **bokeh serve --show myapp**. For the plot to appear, run the bokeh serve from the directory where your myapp folder is placed. For example, If the myapp folder is located inside your dekstop/python directory, traverse (cd desktop/python) to the folder and run the bokeh serve command. This should show the plot in the browser.

The plot will only show the player coordinates at time 0. Now we need to add a slider and vary the time to show the full sequence of event.

    freq = Slider(title="Game Time", value=0, start=all_team.time.unique().min(), end=all_team.time.unique().max(), step=1)


Define the start and end time using **all_team.time.unique().min()/max()**.

Once the slider is created, you'd need to build a function to get the call back when slider is updated. 


    def update_data(attrname, old, new):  
        k = freq.value #holds the current time value of slider after updating the slider
        
        #now again plot x and y based on time on slider
        x = all_team[all_team.time == k].x  
        y = all_team[all_team.time == k].y
        #update the CDS source with new x and y values. player_id and c remains same and it'll be plotted on top of new x and y.
        
        source.data = dict(x=x, y=y,player_id=player_id,color=c)

Everytime the source gets updated, the plots associated with 'source' will get updated, which includes the scatter plot st and labels, since we're using this source to plot those data.
