import pymysql 

connection=pymysql.connect(host="localhost",user="root",database="pfe")
cur=connection.cursor()


cur.execute('select * from produits')
liste=[]
liste2=[]
for row in cur.fetchall():
    liste.append((row[0],row[2],row[-2]))

for i,j,k in liste:
    cur.execute('select type from categories where id=%s'%k)
    for x in cur.fetchall():
        liste2.append(x[0])
print(liste)
print(liste2)

res=list(zip(liste,liste2))

print("*******************************************")
print(res)
liste_id=[]
liste_prd=[]
liste_cat=[]

for i in res:
    liste_id.append(i[0][0])
    liste_prd.append(i[0][1])
    liste_cat.append(i[1])

import pandas as pd

df=pd.DataFrame(data={'product_id':liste_id,"name":liste_prd,"category":liste_cat})
df.to_csv("abir.csv",sep=",",index=False)
print("done")
