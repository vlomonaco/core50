---
title:  "First blog post"
date:   2017-01-19 15:04:23
categories: [blogpost]
tags: [me, first]
---
`Thank` you for being here! :-)

This is just a fake post to test some functionalities. Here you can see how easily is in Python to query google and get the results:

``` python
import urllib2
import urllib
import json

url = "http://ajax.googleapis.com/ajax/services/search/web?v=1.0&"
query = raw_input("What do you want to search for ? >> ")
query = urllib.urlencode( {'q' : query } )
response = urllib2.urlopen (url + query ).read()
data = json.loads ( response )
results = data [ 'responseData' ] [ 'results' ]

for result in results:
    title = result['title']
    url = result['url']
    print ( title + '; ' + url )
```

[jekyll]:      http://jekyllrb.com
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-help]: https://github.com/jekyll/jekyll-help
