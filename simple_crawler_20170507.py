# -*- coding: utf-8 -*-
"""
Created on Sun May  7 15:04:56 2017

前半：　最初に指定したページのURL以下のリンク先URLを取得する。
後半：　そのリンク先URLのHTMLデータを取得する。
"""

import re
import requests
import lxml.html


#前半：　最初に指定したページのURL以下のリンク先URLを取得する。

#　指定ページ
domain01 = 'http://www.yahoo.co.jp/'


def func01(domain):
    response = requests.get(domain)
    root=lxml.html.fromstring(response.content)
    root.make_links_absolute(response.url)
    aa = list()
    cnt00 = 0
    for a in root.xpath('//a'):
        url=a.attrib['href']     #なぜか、一部の <a href=""  >  を拾わない。なぜ？今後の課題
        aa.append(url)
        cnt00+=1
    ls01=list(set(aa))
    ls01=[x for x in ls01 if x is not None]
    ls01=[x for x in ls01 if x.startswith(domain01)]
    ls01.sort()
    return ls01


#行ったリスト
ls_p = list()
#行くべきリスト
ls_f = [domain01]
#全てのページリスト
ls_all = list()

cnt = 0
while len(ls_f)!=0:
    x=ls_f[0]
    ls_p.append(x)
    try:
        ls_all = list(set(ls_p) | set(ls_f) | set(func01(x)) )
    except:
        ls_all = list(set(ls_p) | set(ls_f))
    ls_f = list(set(ls_all)-set(ls_p)) 
    ls_f.sort()      
    cnt += 1
    print(cnt,x)
    
ls_all.sort()    
ls_f.sort()
ls_p.sort()





#後半：　そのリンク先URLのHTMLデータを取得する。


#ls_all のページを回って、HTMLデータを取ってくる。
ls_pages = list()
for x in ls_all:
    r = requests.get(x)    
    ls_pages.append(r.text)

ls_pages02=list(set(ls_pages))
ls_pages02.sort()


from pandas import DataFrame as DF
df01 = DF({"URL":ls_all,"DOC_TEXT":ls_pages})

ls_natu = list()
ls_dup = list()
for i,x in enumerate(ls_pages):
    if x not in ls_natu: 
        ls_natu.append(x)
    else: 
        ls_dup.append(ls_all[i]) 





