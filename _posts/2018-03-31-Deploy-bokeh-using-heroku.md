# Deploy bokeh server plots using Heroku

This is a small blog post, guiding you with steps to deploy a bokeh python server plot using Heroku. There are very little resources online, that outline the process. The documentation is not very clear (least I found it little confusing, when different people suggest different methods) and hopefully this blog provides a clear guideline. 

Assuming you've your serve plot ready to be deployed, let's begin by creating folders that'd be required. My **herokuapp** has the following folders directory.

myapp<br/>
   |<br/>
   +---data **(folder)**<br/>
   |    +---data.csv<br/>
   |<br/>
   +---main.py<br/>
   +---static **(folder)**<br/>
   |    +---images **(folder)**<br/>
   |    |    +---images.png<br/>
   
The same has to be followed while creating repository in github. This is how my repository looks like.

![alt text](https://raw.githubusercontent.com/samirak93/analytics/gh-pages/blog_images/images/blog1/repo.PNG)

This is the easy part as this is how you'd run the myapp locally in your system.

The main part, creating text documents for heroku, is the tricky part. There are various answers onnline regarding this. I'd to trial out few methods before getting it right. 

So the whole **myapp** folder is within my main repository, **herokuapp**. Inside herokuapp, you'd have to create 3 documents. Note that these file names are case sensitive.

  - Procfile	
  - requirements.txt	
  - runtime.txt	
  
### Procfile
```sh
web: bokeh serve --port=$PORT --allow-websocket-origin=heroku_app_name.herokuapp.com --address=0.0.0.0 --use-xheaders myapp
```

### requirements.txt
```sh
bokeh==0.12.15
Jinja2==2.10
MarkupSafe==1.0
numpy==1.14.2
pandas==0.19.2
PyYAML==3.10
requests==2.18.4
scikit-learn==0.19.1
scipy==1.0.1
tornado==5.0.1
```

### runtime.txt
```sh
python-2.7.14
```

