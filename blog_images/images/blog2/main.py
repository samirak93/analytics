import numpy as np
import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import row, widgetbox,column
from bokeh.models import ColumnDataSource,LabelSet,PointDrawTool,CustomJS
from bokeh.models.widgets import Slider,Paragraph,Button,CheckboxButtonGroup
from bokeh.plotting import figure
from scipy.spatial import ConvexHull

df=pd.read_csv('data/data.csv')
headers = ["x", "y", "team_id", "player_id","time"]
all_team = pd.DataFrame(df, columns=headers)

team_att =all_team[all_team.team_id==1]
team_def =all_team[all_team.team_id==2]
ball=all_team[all_team.team_id==3]
i=0

t1 = np.vstack((team_att[team_att.time==i].x, team_att[team_att.time==i].y)).T
t2 = np.vstack((team_def[team_def.time == i].x, team_def[team_def.time == i].y)).T

hull=ConvexHull(t1)
hull2= ConvexHull(t2)
xc = t1[hull.vertices, 0]
yc = t1[hull.vertices, 1]

ax = t2[hull2.vertices, 0]
ay = t2[hull2.vertices, 1]


player_id=['1','2','3','4','5','6','7','8','9','10','11','1','2','3','4','5','6','7','8','9','10','11',' ']
c=['dodgerblue','dodgerblue','dodgerblue','dodgerblue','dodgerblue','dodgerblue','dodgerblue','dodgerblue','dodgerblue','dodgerblue','dodgerblue','orangered','orangered','orangered',
   'orangered','orangered','orangered','orangered','orangered','orangered','orangered','orangered','gold']


source = ColumnDataSource(data=dict(x=x, y=y,player_id=player_id,color=c))
source2 = ColumnDataSource(data=dict(xc=xc,yc=yc))
source3 = ColumnDataSource(data=dict(ax=ax,ay=ay))


plot = figure(name='base',plot_height=600, plot_width=800, title="Game Animation",
              tools="reset,save",
              x_range=[-52.5,52.5], y_range=[-34, 34],toolbar_location="below")
plot.image_url(url=["myapp/static/images/base.png"],x=-52.5,y=-34,w=105,h=68,anchor="bottom_left")

plot.xgrid.grid_line_color = None
plot.ygrid.grid_line_color = None
plot.axis.visible=False

st=plot.scatter('x','y', source=source,size=20,fill_color='color')

labels=LabelSet(x='x', y='y', text='player_id', level='glyph',
              x_offset=-5, y_offset=-7, source=source, render_mode='canvas',text_color='white',text_font_size="10pt")


freq = Slider(title="Game Time", value=0, start=all_team.time.unique().min(), end=all_team.time.unique().max(), step=1)


def update_data(attrname, old, new):


    k = freq.value
    # shadow_draw(k)

    x = all_team[all_team.time == k].x
    y = all_team[all_team.time == k].y
    source.data = dict(x=x, y=y,player_id=player_id,color=c)

    t1 = np.vstack((team_att[team_att.time==k].x, team_att[team_att.time==k].y)).T
    t2 = np.vstack((team_def[team_def.time==k].x, team_def[team_def.time==k].y)).T

    hull = ConvexHull(t1)
    hull2= ConvexHull(t2)
    xc = t1[hull.vertices, 0]
    yc = t1[hull.vertices, 1]

    ax = t2[hull2.vertices, 0]
    ay = t2[hull2.vertices, 1]
    
    source2.data=dict(xc=xc,yc=yc)
    source3.data=dict(ax=ax,ay=ay)


for w in [freq]:
    w.on_change('value', update_data)

plot.add_layout(labels)

button = Button(label='► Play', width=60)

def animate_update():
    year = freq.value + 1
    if year > all_team.time.max():
        year = all_team.time[0]
    freq.value = year
    
def animate():
    if button.label == '► Play':
        button.label = '❚❚ Pause'
        curdoc().add_periodic_callback(animate_update, 50)
    else:
        button.label = '► Play'
        curdoc().remove_periodic_callback(animate_update)

button.on_click(animate)

team_Blue=plot.patch('xc', 'yc', source=source2, alpha=0, line_width=3, fill_color='dodgerblue')
team_red = plot.patch('ax', 'ay',source=source3, alpha=0, line_width=3,fill_color='orangered')

checkbox_blue=CheckboxButtonGroup(labels=["Team Blue"],button_type = "primary")
checkbox_red=CheckboxButtonGroup(labels=["Team Red"],button_type = "primary")

checkbox_blue.callback = CustomJS(args=dict(l0=team_Blue, checkbox=checkbox_blue), code="""
l0.visible = 0 in checkbox.active;
l0.glyph.fill_alpha = 0.3;
""")
checkbox_red.callback = CustomJS(args=dict(l0=team_red, checkbox=checkbox_red), code="""
l0.visible = 0 in checkbox.active;
l0.glyph.fill_alpha = 0.3;
""")

p = Paragraph(text="""Select team  to plot convex hull""",
width=200)

inputs = widgetbox(freq,button,p,checkbox_blue,checkbox_red)

layout = column(row(inputs,plot))

curdoc().add_root(layout)

curdoc().title = "Game Animation"
