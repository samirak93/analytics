---
layout: post
title: 'Where do teams cross?'
---

This post looks at building an interactive tool, using python, to find out the end locations of crosses based on user input (cross start location). The user input gets the cross start location. Using that, the crosses nearby the start location are identified and their corresponding end locations are plotted on the graph. This uses a simple concept called <b>cKDTree</b>, which uses nearest-neighbor lookup concept.


<b>Application:

When trying to create this interactive graph, I wanted to look at this from a coach/analyst perspective. Ideally, people would want to see where successful crosses <i>(crosses that either lead to a goal/ shot at goal)</i> end up. These are usually represented by graphs similar to below.

![alt text](https://raw.githubusercontent.com/samirak93/analytics-blog/master/_posts/images/heat_map.jpg)

<i>Crosses Heat Map. (Credits: Samirak93)</i>



But these images usually depict the entire dataset of crosses that are analysed. If filtered down further, they are usually  filtered down with respect to a particular player/team.

I wanted to narrow this same image to the point where these images are rendered based on user interactivity. 
So if the user clicks on a particular location on the pitch, the crosses nearest to it are collected and their corresponding end locations are shown. This is much more useful when user wants to visualise crosses only from certain locations rather than getting a complete view.

So the concept of [cKDTree](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html) comes into picture here. It's relatively simple to use.

The input here is the cross start location X and Y coordinates. One of the parameters we can manually provide is the search radius of the KDTree. In this case, the search radius given is 6 units.So the system find the nearest crosses around 6 units.

<b>"point_tree.query_ball_point"</b> is the function to find the cross_end X and Y coordinates. 


The entire code is made available here: https://github.com/samirak93/Where-do-teams-cross-/blob/master/crosses_KD.py

I tried to build this as a working demo (using python and d3js) but couldn't come up with anything solid. If you can bring this as a demo online, I'd be delighted to get a look at it. 

If you've any comments/issues on the post/code, please do let me know. I'm a beginner in python and please do expect any flaws in the code :)

<b>Note:</b>The data used here belongs to Opta and hence are not made available in the above code link.

Leave your comments below or contact me on twitter: [@SamiraK93](https://twitter.com/Samirak93)
