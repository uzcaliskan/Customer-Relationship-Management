# Customer Life Time Value - CLTV (Müşteri Yaşam Boyu Değeri):
# Bir müşterinin bir şirketle kurduğu ilişki-iletişim süresince bu şirkete kazandıracağı parasal değerdir.
# Bir müşterinin işletme için üreteceği bütün gelirlerin beklenen değeri/toplamı
# Nasıl Hesaplanır ?
# Literatürde birçok yolu olduğu söylenmektedir. Temel olarak ise:
# CLTV(Customer LifeTime Value)= (Customer Value / Churn Rate) * Profit Margin
# CV (Customer Value) = satın alma başına ortalama kazanç * satın alma sayısı
# CV (Customer Value) = Average Order Value * Purchase Frequency
# Not: Customer Value Çok önelidir.
# Average Order Value = Total Price / Total Transaction
# Purchase Frequency = Total Transaction / Total Number of Customers
# Churn Rate = 1 – Repat rate
# Müşteri terk oranı = 1 – müşteri elde tutma oranı(repeat rate)
# Müşteri elde tutma oranı (repeat rate) = birden fazla alışveriş yapan müşteriler / tüm müşteriler
# Profit Margin = Total Price * 0,10
# Transaction: İşlem(Bizim için fatura kesilme işlemi)
#
# Sonuç olarak: Her bir müşteri için hesaplancak olan CLTV değerlerine göre bir sıralama yapıldığında ve CLTV değerlerine göre belirli noktalardan bölme işlemi yapılarak gruplar oluşturulduğunda müşterimiz Segmentlere ayrılacaktır!

import pandas as pd, numpy as np
pd.set_option("display.max_columns", None)
# pd.set_option("display.max_rows", None)
pd.set_option("display.width", 1000)
pd.set_option("display.float_format", lambda x: "%.5f" % x) # float sayıların virgülden sonra 5 basamağını da göstermesi ayarlanıyor!
df_ = pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2009-2010")
df = df_.copy()
df.head()
# 1. Veriyi Hazırlama
from sklearn.preprocessing import MinMaxScaler
df = df[~df["Invoice"].str.contains("C", na=False)]
df.dropna(subset= "Customer ID", inplace=True)  # sadece customerid kolonundaki boşları çıkardı.
df.dropna(inplace=True)
df.shape
df = df[df["Quantity"] > 0]
df.describe().T

df["TotalPrice"] = df["Quantity"] * df["Price"]
df.head()

cltv_c = df.groupby("Customer ID").agg({"Invoice": lambda x: x.nunique(),
                                        "Quantity": lambda x: x.sum(),
                                        "TotalPrice": lambda x: x.sum()})
#cltv_c alternatifi:
# cltv_c = df.groupby("Customer ID").agg({"Invoice": lambda x: x.nunique(),
#                                         "Quantity": "sum",
#                                         "TotalPrice": "sum"})
cltv_c.columns = ["total_transaction", "total_unit", "total_price"]


# 2. Average Order Value(Oratalama Sipariş Değeri) = Total Price / Total Transaction
cltv_c.head()
cltv_c["average_order_value"] = cltv_c["total_price"] / cltv_c["total_transaction"]

#3. Purchaase Frequency (Satın Alma Sıklığı) = Total Transaction / Total Number of Customers

cltv_c["purchase_frequency"] = cltv_c["total_transaction"] / cltv_c.shape[0]


#4. Tekrarlama Oranı & Müşteri Terki(kaybetme) Oranı (Repeat rate & churn Rate) = birden fazla alışveriş yapan müşteriler / tüm müşteriler

repeat_rate = cltv_c[cltv_c["total_transaction"] > 1].shape[0] / cltv_c.shape[0]
churn_rate = 1 - repeat_rate

# profit margin - kar margin. Profit Margin = Total Price * 0,10. not: buradaki 0.10 değeri şirket tarafından belirlenmesi gereken bir değerdir.
cltv_c["profit_margin"] = cltv_c["total_price"] * 0.10

#customer value - müşteri değeri  (Customer Value) = Average Order Value * Purchase Frequency
cltv_c["customer_value"] = cltv_c["average_order_value"] * cltv_c["purchase_frequency"]

# customer lifetime value - müşteri yaşamboyu değeri = Customer Value / Churn Rate * Profit Margin

cltv_c["cltv"] = (cltv_c["customer_value"] / churn_rate) * cltv_c["profit_margin"]
cltv_c.sort_values(by="cltv", ascending=False)

#müşterilerin segmetlere ayrılması
cltv_c.sort_values(by="cltv", ascending=False)
cltv_c["segment"] = pd.qcut(cltv_c["cltv"], 4, labels=["D", "C", "B", "A"])
cltv_c.groupby("segment").agg({"count", "mean", "sum"})
cltv_c.to_csv("cltv_c.csv")

### Tüm işlemlerin fonksitonlaştırması
def create_cltv_c(dataframe, profit=0.10):
    #veri hazırlama
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    cltv_c = dataframe.groupby("Customer ID").agg({"Invoice": lambda x: x.nunique(),
                                            "Quantity": lambda x: x.sum(),
                                            "TotalPrice": lambda x: x.sum()})
    cltv_c.columns = ["total_transaction", "total_unit", "total_price"]
    # 2. Average Order Value(Oratalama Sipariş Değeri) = Total Price / Total Transaction
    cltv_c["average_order_value"] = cltv_c["total_price"] / cltv_c["total_transaction"]
    # 3. Purchaase Frequency (Satın Alma Sıklığı) = Total Transaction / Total Number of Customers
    cltv_c["purchase_frequency"] = cltv_c["total_transaction"] / cltv_c.shape[0]
    # 4. Tekrarlama Oranı & Müşteri Terki(kaybetme) Oranı (Repeat rate & churn Rate) = birden fazla alışveriş yapan müşteriler / tüm müşteriler
    repeat_rate = cltv_c[cltv_c["total_transaction"] > 1].shape[0] / cltv_c.shape[0]
    churn_rate = 1 - repeat_rate
    # profit margin - kar margin. Profit Margin = Total Price * 0,10. not: buradaki 0.10 değeri şirket tarafından belirlenmesi gereken bir değerdir.
    cltv_c["profit_margin"] = cltv_c["total_price"] * profit
    # customer value - müşteri değeri  (Customer Value) = Average Order Value * Purchase Frequency
    cltv_c["customer_value"] = cltv_c["average_order_value"] * cltv_c["purchase_frequency"]
    # customer lifetime value - müşteri yaşamboyu değeri = Customer Value / Churn Rate * Profit Margin
    cltv_c["cltv"] = (cltv_c["customer_value"] / churn_rate) * cltv_c["profit_margin"]
    #segmenlerin oluştuurlması
    cltv_c["segment"] = pd.qcut(cltv_c["cltv"], 4, labels=["D", "C", "B", "A"])
    return cltv_c


create_cltv_c(df)


