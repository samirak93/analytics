<b><h>Where do teams cross?</h></b>

This post looks at building an interactive tool, using python, to find out the end locations of crosses based on user input (cross start location). The user input gets the cross start location. Using that, the crosses nearby the start location are identified and their corresponding end locations are plotted on the graph. This uses a simple concept called cKDTree, which uses nearest-neighbor lookup concept.


Application:

When trying to create this interactive graph, I wanted to have look at this from a coach/analyst perspective. Ideally, people would want to see where successful crosses (crosses that either lead to a goal/ shot at goal) end up. These are usually represented by graphs similar to below.

![alt text](https://raw.githubusercontent.com/samirak93/analytics-blog/master/_posts/images/heat_map.jpg)

<i>Crosses Heat Map. (Credits: Samirak93)</i>



But these images usually depict the entire dataset of crosses that are analysed. If filtered down further, they can be filtered down with respect to a particular player/team.

I wanted to narrow this same image to the point where these images are rendered based on user interactivity. So if the user clicks on a particular location on the pitch, the crosses nearest to it are collected and their corresponding end locations are shown. This is much more useful when user wants to visualise crosses only from certain locations rather than getting a complete view.

The entire code is made available here: https://github.com/samirak93/Crosses_KD

The data used here belong to Opta and are not made available in the above link.










Created by Samira Kumar 
Twitter: @Samirak93

Licence: Free to use/ modify
