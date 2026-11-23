import requests, json
U="https://scanner.tradingview.com/egypt/scan"; H={"User-Agent":"Mozilla/5.0"}
names=["AIHC","UASG","ACRO","ESRS","ARVA","HCFI","MKIT","NAPR","DIFC","FCMD"]
b={"columns":["name","description","type","subtype","close","time","exchange"],"range":[0,50],"markets":["egypt"],
   "filter":[{"left":"name","operation":"in_range","right":names}]}
print(requests.post(U,json=b,headers=H,timeout=20).text[:3000])
b={"columns":["name","description","type","close"],"range":[0,50],"markets":["egypt"],
   "filter":[{"left":"description","operation":"match","right":"Ezz"}]}
print(requests.post(U,json=b,headers=H,timeout=20).text[:3000])
b={"columns":["name","type"],"range":[0,2000],"markets":["egypt"]}
d=requests.post(U,json=b,headers=H,timeout=20).json(); print(d["totalCount"]); from collections import Counter; print(Counter(x["d"][1] for x in d["data"]))
