### RFM ANALİZİ
# RFM - Recency, Frequency, Monetary (En sonki müşteri satın alması, müşterinin ne kadar ürün satın aldığı, müşterinin bıraktığı para)
# RFM analizi müşteri segmentasyonu için kullanılan bir tekniktir. Müşterilerin satın alma alışkanlıkları üzerinden gruplara ayrılması ve bu gruplar üzerinde strateji geliştirilmesini sağlar.
# CRM çalışmaları kapsamında veriye dayalı aksiyon alınmasını sağlar.
# RFM Metrikleri  RFM skorları  Müşteri Segmentasyonu
# RFM Metrikleri:
# Recency (Yenilik): Müşterinin bizden en son ne zaman alışveriş yaptığını gösteriyor. Ne kadar düşükse o kadar iyidir.
# Frequency (Sıklık): Müşterinin Yaptığı toplam alışveriş / işlem sayısıdır.
# Monetary (Parasal Değer): Müşterinin bize bıraktığı toplam para miktarıdır.
# NOT: CRM çalışmalarında frequency monetary den daha önemlidir. Çünük daha fazla sıklığı olan müşteri toplamda daha fazla getiri getirecektir.


# 1. İş Problemi:
# 2. Veriyi Anlama (Data Understanding)
# 3. Veri Hazırlama ( Data preparation)
# 4. RFM Metriklerinin Hesaplanması
# 5. RFM Sokorlarının Hesaplanması
#6. RFM Segmentlerinin oluşturulması ve analiz edilmesi.( Creating & Analysing RFM Segments)
#7. Tüm Sürecin Fonksiyonlaştırılması


#1. İş Problemi:
# Bir e ticaret şirketi müşterilerini segmetnlere ayırıp bu segmentlere göre pazarlama stratejisi belirlemek istiyor.
# Bir e-ticaret şirketi müşterilerini segmentlere ayırıp bu segmentlere göre
# pazarlama stratejileri belirlemek istiyor.

# Veri Seti Hikayesi
# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

# Online Retail II isimli veri seti İngiltere merkezli online bir satış mağazasının
# 01/12/2009 - 09/12/2011 tarihleri arasındaki satışlarını içeriyor.

# Değişkenler
#
# InvoiceNo: Fatura numarası. Her işleme yani faturaya ait eşsiz numara. C ile başlıyorsa iptal edilen işlem.
# StockCode: Ürün kodu. Her bir ürün için eşsiz numara.
# Description: Ürün ismi
# Quantity: Ürün adedi. Faturalardaki ürünlerden kaçar tane satıldığını ifade etmektedir.
# InvoiceDate: Fatura tarihi ve zamanı.
# UnitPrice: Ürün fiyatı (Sterlin cinsinden)
# CustomerID: Eşsiz müşteri numarası
# Country: Ülke ismi. Müşterinin yaşadığı ülke.

# 2. Veriyi Anlama (Data Understanding)
import pandas as pd, numpy as np
pd.set_option("display.max_columns", None)
# pd.set_option("display.max_rows", None)
pd.set_option("display.width", 1000)
pd.set_option("display.float_format", lambda x: "%.3f" % x) # float sayıların virgülden sonra 3 basamağını da göstermesi ayarlanıyor!
df_ = pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2009-2010")
df = df_.copy()
df.head(100)
df.shape
df.info()
df.isnull().sum()
df.columns = df.columns.str.lower()
df["description"].nunique()
df["invoice"].nunique()
df["description"].value_counts().head() # kaç farklı ürün var
df.groupby("description").agg({"quantity" : "sum"}).sort_values("quantity", ascending=False) # en fazla hangi üründen satılmış
df.isnull().sum()
df["invoice"].nunique() # kaç farklı fatura kesilmiş
#### 3. veriyi hazırlama


df["total_price"] = df["quantity"] * df["price"]
df.groupby("invoice").agg({"total_price": "sum"}).sort_values("total_price", ascending=False) # hangi faturadan toplamda ne kadar kazanılmış
df.isnull().sum()
df.dropna(inplace=True) # verideki customerid değişkeninde eksiklik olduğundan boş değerleri sildik !
#df.dropna(subset = ['column_name']) # specific bir kolondaki değerleri silmek
df.shape
df.head()
df = df[~df["invoice"].str.contains("C", na=False)] # contains metodunda na=False kullanılarak boş değerler False olarak algılanıyor/yazılıyor
# df[~df["invoice"].str.contains("C", na=False)] ile fatura kısmında iptal edilen(başında C olan) kısımlar df in dışında bırakılıyor!
df.describe().T

# 4. RFM metriklerinin hesaplanması - Recency, Frequency, Monetary - calculating rfm metrics
df.head()
df["invoicedate"].max() #çıktı: Timestamp('2010-12-09 20:01:00') . analiz tarihinini en sonki alışveriş tarihini bulup üstüne 2 gün ekledikten sonra buluyoruz.
import datetime as dt
today_date = dt.datetime(2010, 12, 11)
type(today_date)
rfm = df.groupby("customer id").agg({"invoicedate": lambda date: (today_date - date.max()).days,  #Recency - bugun ile en sonki alışveriş arasındaki fark
                               "invoice" : lambda num: num.nunique(),                     #Kaç farklı fatura kesilmiş - frequency - kaç farklı alışveriş yapılmış
                               "total_price": lambda total: total.sum()})                                 #müşterinin toplam bıraktığı para
rfm.columns = ["recency", "frequency", "monetary"]
rfm.head()
rfm.describe().T
rfm = rfm[rfm["monetary"] > 0] ## parasal değeri 0 dan büyük olanları seçiyoruz.

rfm.head()
rfm.shape

#5 - rfm skorlarının hesaplanması - calculating rfm scores

rfm.head()
rfm.columns
rfm["recency_score"] = pd.qcut(rfm.loc[:, "recency"], 5, labels=[5, 4, 3, 2, 1])
#qcut: quntile(çeyreklik) değerlerine göre bölme yapar. öncelikli veriyi küçükten büyüğe sıralar ve ilgili etiketleri ilgili aralığıa koyar
# recency - müşterinin alışveriş yaptığı en sonki günlük süre- ne kadar küçükse bizim için o kadar iyi olduğundan en küçük recency
# değerine en büyük recency skoru(5) etiket olarak verildi !!1
rfm.head()
rfm["monetary_score"] =  pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])
# yuarıdaki labels kısmında büyük görülen parçalar büyük, küçük görülen parçalar ise küçük sayıyla yazılacka !

rfm["frequency_score"] = pd.qcut(rfm["frequency"], 5, labels=[1, 2, 3, 4, 5])
# yukarıdaki kod ile uyıar alınıyor :You can drop duplicate edges by setting the 'duplicates' kwarg
# yani 5 parçaya bölünen çeyreklik değerlerde örneğin; ilk çeyreklik ile ikinci çeyreklik değerlerinde çok sayısa aynı sayıdan var.
#bu durumda rank metodu ile, rank(method="first") kullanılarak ilgili değişken sıralanarak çok sayıdaki değerler hangi sıralamadaysa o sıralamayı koruyarak parçalara bölünüyor.
# yani rank(method="first") ile qcut ın ilk gördüğü değerler ilk sıralamaya alınıyor
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm.head()
rfm["RFM_score"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)

#6. RFM segmentlerinin oluşturulması ve Analiz Edilmesi (creating and Analysing RFM segments)
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}
seg_map = {r"[1-2][1-2]" : "hibernating",
           r"[1-2][3-4]" : "at_risk",
           r"[1-2]5" : "cant_loose",
           r"3[1-2]" : "about_to_sleep",
           r"33" : "need_attention",
           r"[3-4][4-5]" : "loyal_customers",
           r"41" : "promissing",
           r"51" : "new_customers",
           r"[4-5][2-3]" : "potential_loyalists",
           r"5[4-5]" : "champions"
            }

# r"[1-2][3-4]" : "at_risk"   r ifaredesi regex in kısaltması. [] her bir elemanı ifade ediyor. buradaki ifade ilk elamnında 1 ya da 2 geçen ve ikinci elemanında 3 ya da 4
# geçen ifadelere "at_risk" yazdırılıyor.
rfm["segment"] = rfm["RFM_score"].replace(seg_map,regex=True)
rfm[["segment", "recency", "frequency", "monetary" ]].groupby("segment").agg(["mean", "count"])
rfm[rfm["segment"] == "need_attention"]

new_df = pd.DataFrame()

new_df["new_customer_id"] = rfm[rfm["segment"] == "new_customers"].index
new_df["new_customer_id"] = new_df["new_customer_id"].astype(int)
new_df.to_csv("new_customers.csv")
rfm.to_csv("rfm.csv")

#### 7. TÜM SÜRECİN FONKSİYONLAŞTIRLMASI
def create_rfm(dataframe, csv=False):
    # veriyi hazırlama
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]  # contains metodunda na=False kullanılarak boş değerler False olarak algılanıyor/yazılıyor
    # RFM Metriklerinin Hesaplanması
    import datetime as dt
    today_date = dt.datetime(2011, 12, 11)
    rfm = df.groupby("Customer ID").agg({"InvoiceDate": lambda date: (today_date - date.max()).days,
                                         # Recency - bugun ile en sonki alışveriş arasındaki fark
                                         "Invoice": lambda num: num.nunique(),
                                         # Kaç farklı fatura kesilmiş - frequency - kaç farklı alışveriş yapılmış
                                         "TotalPrice": lambda total: total.sum()})
                                         # müşterinin toplam bıraktığı para
    rfm.columns = ["recency", "frequency", "monetary"]
    rfm = rfm[rfm["monetary"] > 0]  ## parasal değeri 0 dan büyük olanları seçiyoruz.
    rfm["recency_score"] = pd.qcut(rfm.loc[:, "recency"], 5, labels=[5, 4, 3, 2, 1])
    rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["RFM_score"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)
    seg_map = {r"[1-2][1-2]": "hibernating",
               r"[1-2][3-4]": "at_risk",
               r"[1-2]5": "cant_loose",
               r"3[1-2]": "about_to_sleep",
               r"33": "need_attention",
               r"[3-4][4-5]": "loyal_customers",
               r"41": "promissing",
               r"51": "new_customers",
               r"[4-5][2-3]": "potential_loyalists",
               r"5[4-5]": "champions"
               }
    rfm["segment"] = rfm["RFM_score"].replace(seg_map, regex=True)
    rfm = rfm[["segment", "recency", "frequency", "monetary"]]
    rfm.index.astype(int)
    if csv:
        rfm.to_csv("rfm.csv")
    return rfm
new_rfm = create_rfm(df, csv=True)

