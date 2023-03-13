# Zaman Projeksiyonlu olasılıksal Life Time Value Tahmini: Yani müşterinin şirketle kurduğu ilişki dönemi boyunca toplamda ne kadar para getireceğinin tahminini yapıyoruz.
# CV (Customer Value) = satın alma başına ortalama kazanç * satın alma sayısı
# Not: Yukarıdaki satın alma başına ortalama kazanç müşterinin toplam yaptığı alışverişlerden elde edilirken satın alma sayısı belirli bir periyot için düşünülebilir. Örneğin bir müşteri toplamda 30 birimlik 10 farklı alışveriş yaptıysa bu müşterinin aylık bazda getireceği alışveriş bu satın alma başına gerçekleştirilen * aylık ortalama alışveriş sayısı ile bulunuyor.
# Olasılıksal bazda bir müşterinin şirketle ilişkisi boyunca ne kadar getireceğini hesaplamak için;
# CLTV = Conditional Expected Number of Transaction * Conditional Expected Average Profit
# CV (Customer Value) = Purchase Frequency * Average Order Value
# Not: Purchase = Transaction
# Conditional Expected Number of Transaction: Bütün kitlenin satın alma davranışını bir olasılık dağılımıyla modellendikten sonra, bu genel satın alma davranışını kişi özelinde(conditional) biçimlendirecek şekilde kullanarak her bir müşteri için  beklenen satın alma / işlem sayıları tahmin edilir.
# Conditional Expected Average Profit: Bütün kitlenin Average Profit değerini olasılıksal olarak modelledikten sonra, bu modele kişi özelliklerini vererek, kişilerin özelinde Conditional Expected Average Profit değerleri hesaplanır.
# CV (Customer Value) = Purchase Frequency * Average Order Value
# CLTV = Conditional Expected Number of Transaction * Conditional Expected Average Profit
# Conditional Expected Number of Transaction: BG-NBD modeli ile hesaplanıyor
# Conditional Expected Average Profit: Gamma Gamma Submodel ile hesaplanıyor.
# CLTV =  BG-NBD modeli * Gamma Gamma Submodel
# BG / NBD ( Beta Geometric / Negative Binomial Distribution) ile Expected Number of Transaction: ( Expected Number of Transaction = Expected Sales Forecasting)
#
# CLTV = Conditional Expected Number of Transaction * Conditional Expected Average Profit
# CLTV =  BG-NBD modeli * Gamma Gamma Submodel
# Expected: Bir rastsal değişkenin beklenen değerini(ortalamasını) ifade eder.
# Rastsal Değişken: Değerlerini bir deneyin sonuçlarından alan değişkene rastsal değişken denir. Bir değişkenin belirli bir olasılık dağılımı izlediğini varsaydığımızda bu değişkenin ortalamasına rastsal değişken denir.
# BG-NBD Modeli literatürde buy till you die olarak da tanımlanmaktadır. Buy till you die demek, öncelikle satın alma davranışı sergilemek ve sonrasında bu süreci sonlandırarak drop, churn olmak (till you die) anlamına gelir.
# BG-NBD Modeli, Expected Number of Transaction için iki süreci olasılıksal olarak modeller.
# 1.	Transaction Process ( Buy)
# 2.	Dropout Process – inaktif olma süreci (Till you die)
# Transaction Process(Buy):
# •	Alive(Canlı) olduğu sürece, belirli bir zaman periyodunda, bir müşteri tarafından gerçekleştirilecek işşlem sayısı transaction rate parametresi ile possion dağılır.
# •	Bir müşteri Alive olduğu sürece (Dropout olmadığı sürece) kendi transaction rate’i etrafında rastgele satın alma yapmaya devam edecektir.
# •	Transaction rate’ler her bir müşteriye göre değişir ve tüm kitle için gamma dağılır (r,a).
# Dropout Process (Till you die):
# •	Her bir müşterinin p olasılığı ile dropout rate (dropout probability)’i vardır.
# •	Bir müşteri alışveriş yaptıkan sonra belirli bir olasılıkla drop olur.
# •	Dropout rate’ler her bir müşteriye göre değişri ve tüm kitle için beta dağılır(a, b).
# Gamma Gamma Submodel
# Bir müşterinin işlem başına ortalama ne kadar kar getirebileceğini tahmin etmek için kullanılır.
# CLTV = Conditional Expected Number of Transaction * Conditional Expected Average Profit
# CLTV =  BG-NBD modeli * Gamma Gamma Submodel
# •	Bir müşterinin işlemlerinin parasal değeri (monetary) transaction value’ların ortalaması etrafında rastgele dağılır.
# •	Ortalama transaction value, zaman içinde kullanıcılar arasında değişebilir fakat tek bir kullanıcı için değişmez. ?
# •	Ortalama transaction value tüm müşteriler arasında gamma dağılır.
import numpy as np, numpy as np
# BG-NBD ve Gamma Gamma ile CLTV Tahmini:
# 1. Verinin Hazırlanması:
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import datetime as dt
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler

df_ = pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2010-2011")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.float_format", lambda x: "%.4f" % x)  # virgünden sonra 5 basamak gösterilecek!
df = df_.copy()
df.head()

# Aşağıdaki fonksiyon ile aykırı değerler için alt ve üst limit belirlenecek fonksiyon yazılmıştır.
#quantile değerlerindeki 0.01 değeri alt limit için, 0.99 değeri ise üst limit için yorumsal olarak verilmiştir.
# normalde low limit için 0.25 uplimit için 0.75 kullanılır
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
# aşağıdaki fonksiyon ile low limit ve up limit üzerindeki değişken değerleri(outlier / aykırı değer) low limit ve
# up limit e set edilerek aykırı değerler baskılanıyor !

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit # veri setini okuttuktan sonra alt limit aykırı dğeerler silineceğinden bunu kapattık
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
###Veri Önişleme
df.head()
df.isnull().sum()
df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df.shape
df.describe().T

df = df[df["Price"] > 0]
df = df[df["Quantity"] > 0]
replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")
df["TotalPrice"] = df["Price"] * df["Quantity"]
dt.datetime.date((df["InvoiceDate"].max())).year
dt.datetime.date((df["InvoiceDate"].max())).month
dt.datetime.date((df["InvoiceDate"].max())).day + 2
#dt.datetime(yıl, ay, gün)
today_date = dt.datetime(dt.datetime.date((df["InvoiceDate"].max())).year, dt.datetime.date((df["InvoiceDate"].max())).month, dt.datetime.date((df["InvoiceDate"].max())).day + 2)

# lifetime Veri Yapısının Hazırlanması
#BG-NBD(BetaGeoFitter) ve Gamma Gamma modellerinin (GammaGammaFitter) İstediği veri formatlarının hazırlanması
# recency: Son satın alma üzerinden geçen zaman. Haftalık. (kullanıcı özelinde) -  müşterinin kendi içinde son satın alma zamanı - ilk satın alma zamanını ifade eder
# T: Müşterinin yaşı. Haftalık. (analiz tarihinden ne kadar süre önce ilk satın alma yapılmış)
# frequency: tekrar eden toplam satın alma sayısı (frequency>1)
# monetary: satın alma başına ortalama kazanç
# not: buradaki recency değeri ile monetary değeri RFM analizindeki recency ve monetary değerinden farklıdır.
df.head()
cltv_df = df.groupby("Customer ID").agg({"InvoiceDate": [lambda x: (x.max() - x.min()).days,
                                               lambda x: (today_date - x.min()).days],
                               "Invoice": lambda x: x.nunique(),
                               "TotalPrice": lambda x: x.sum()
                               })
# cltv_df in sütunlarını şu şekilde : MultiIndex([('InvoiceDate', '<lambda_0>'),
#             ('InvoiceDate', '<lambda_1>'),
#             (    'Invoice',   '<lambda>'),
#             ( 'TotalPrice',   '<lambda>')],
#            )
cltv_df.columns =  cltv_df.columns.droplevel(1) # droplevel(0) en üst seviye, droplevel(1) ile de ikinci seviye silinir
#droplevel ile cltv_df in sütunları şu şekilde: Index(['InvoiceDate', 'InvoiceDate', 'Invoice', 'TotalPrice'], dtype='object')
cltv_df.columns = ["recency", "T", "frequency",  "monetary"]
cltv_df.head()

# ÇOK ÖNEMLİ NOT: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.

# monetary: satın alma başına ortalama kazanç
cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
cltv_df.describe().T
cltv_df = cltv_df[cltv_df["frequency"] > 1] # frequency: tekrar eden toplam satın alma sayısı (frequency>1)

cltv_df["recency"] = cltv_df["recency"] / 7 # recency değeri haftalık olmalıdır !
cltv_df["T"] = cltv_df["T"] / 7 # T - müşteri yaşı değeri haftalık olmalıdır !
cltv_df.head()
#############################################################################
# BG-NBD Modelinin Kurulması:
#############################################################################

bgf = BetaGeoFitter(penalizer_coef=0.001)
#penalizer_coef parametrelerin bulunması esnasında paramaetrelere uygulanacak ceza katsayısıdır.
bgf.fit(cltv_df["frequency"],
        cltv_df["recency"],
        cltv_df["T"])

################################################################################################
# 1 hafta içinde en çok satın alma beklediğimiz 10 müşteri kimdir ?
#############################################################################################

bgf.conditional_expected_number_of_purchases_up_to_time(1, cltv_df["frequency"], cltv_df["recency"], cltv_df["T"]).sort_values(ascending=False)
# buradaki 1, 1 haftayı ifade eder.
bgf.predict(1, cltv_df["frequency"], cltv_df["recency"], cltv_df["T"]).sort_values(ascending=False)

# predict metodu ile conditional_expected_number_of_purchases_up_to_time metodu aynıdır.
cltv_df["expected_purc_1_week"] = bgf.conditional_expected_number_of_purchases_up_to_time(1, cltv_df["frequency"], cltv_df["recency"], cltv_df["T"])

cltv_df.head()
################################################################################################
# 1 ay içinde en çok satın alma beklediğimiz 10 müşteri kimdir ?
#############################################################################################
cltv_df["expected_purc_1_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4, cltv_df["frequency"], cltv_df["recency"], cltv_df["T"])
## NOT: bgf ve conditional_expected_number_of_purchases_up_to_time ile beklenen satış sayısı tahmin ediliyor !
################################################################################################
## Tahmin sonuçlarının değerlendirilmesi  ??
#############################################################################################

plot_period_transactions(bgf)
plt.show(block=True)
################################################################################################
## Gamma Gamma modelinin kuurlması
#############################################################################################

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df["frequency"], cltv_df["monetary"])
cltv_df.head()
ggf.conditional_expected_average_profit(cltv_df["frequency"], cltv_df["monetary"]).sort_values(ascending=False)
cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df["frequency"], cltv_df["monetary"])
cltv_df.sort_values(by="expected_average_profit",ascending=False)

################################################################################################
## BG-NBD ve GG modeli ile CLTV(expected) nin hesaplanması - müşteri yaşam boyu tahmini
#############################################################################################
cltv_df.head()

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df["frequency"],
                                   cltv_df["recency"],
                                   cltv_df["T"],
                                   cltv_df["monetary"],
                                   time=3, # 3 aylık. time ay cinsinden
                                   freq="W", #T'nin frekans bilgisi
                                   discount_rate=0.01 # the monthly adjusted discount rate
                                                         )
cltv = ggf.customer_lifetime_value(bgf, cltv_df["frequency"], cltv_df["recency"], cltv_df["T"], cltv_df["monetary"],
                                              time=3, # 3 aylık. time ay cinsinden
                                              freq="W", #T'nin frekans bilgisi hangi zaman cinsinden
                                              discount_rate=0.01 # the monthly adjusted discount rate
                                    )

cltv.reset_index()
cltv_final = cltv_df.merge(cltv,how="left", on="Customer ID")
cltv_final.sort_values(by="clv", ascending=False)
#3 aylık periyotta yaılması beklene satış sayısı
cltv_df["expected_purc_3_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4*3, cltv_df["frequency"], cltv_df["recency"], cltv_df["T"])

# NOT: BG-NDB (buy till you die) teorisine göre düzenli ortalama bir alışveriş yapan bir müşteri eğer churn/dropout olmadıysa, müşterinin recency değeri arttıkça o müşterinin satın alma ihtimali artıyordur.

################################################################################################
## 5. CLTV ye göre müşterileri segmetleree ayırma
#############################################################################################

cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])
cltv_final = cltv_final.sort_values(by="clv", ascending=False)
cltv_final.head(50)
cltv_final.groupby("segment").agg({"count", "mean", "sum"})

### Tüm çalışmanın Fonksiyonlaştırılması:
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
# aşağıdaki fonksiyon ile low limit ve up limit üzerindeki değişken değerleri(outlier / aykırı değer) low limit ve
# up limit e set edilerek aykırı değerler baskılanıyor !

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit # veri setini okuttuktan sonra alt limit aykırı dğeerler silineceğinden bunu kapattık
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def create_cltv_p(dataframe, month=3):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Price"] > 0]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    dataframe["TotalPrice"] = dataframe["Price"] * dataframe["Quantity"]
    dt.datetime.date((dataframe["InvoiceDate"].max())).year
    dt.datetime.date((dataframe["InvoiceDate"].max())).month
    dt.datetime.date((dataframe["InvoiceDate"].max())).day + 2
    # dt.datetime(yıl, ay, gün)
    today_date = dt.datetime(dt.datetime.date((dataframe["InvoiceDate"].max())).year,
                             dt.datetime.date((dataframe["InvoiceDate"].max())).month,
                             dt.datetime.date((dataframe["InvoiceDate"].max())).day + 2)
    #today_date en sonki tarihten 2 gün sonrası olarak ayarlanıyor ! 2011 verisi olduğu için...
    # recency: Son satın alma üzerinden geçen zaman. Haftalık. (kullanıcı özelinde) -  müşterinin kendi içinde son satın alma zamanı - ilk satın alma zamanını ifade eder
    # T: Müşterinin yaşı. Haftalık. (analiz tarihinden ne kadar süre önce ilk satın alma yapılmış)
    # frequency: tekrar eden toplam satın alma sayısı (frequency>1)
    # monetary: satın alma başına ortalama kazanç
    cltv_df = dataframe.groupby("Customer ID").agg({"InvoiceDate": [lambda x: (x.max() - x.min()).days,
                                                             lambda x: (today_date - x.min()).days],
                                             "Invoice": lambda x: x.nunique(),
                                             "TotalPrice": lambda x: x.sum()
                                             })
    cltv_df.columns = cltv_df.columns.droplevel(1)  # droplevel(0) en üst seviye, droplevel(1) ile de ikinci seviye silinir
    cltv_df.columns = ["recency", "T", "frequency", "monetary"]
    cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"] #monetary: satın alma başına toplam ortalama kazanç
    cltv_df = cltv_df[cltv_df["frequency"] > 1]  # frequency: tekrar eden toplam satın alma sayısı (frequency>1)
    cltv_df["recency"] = cltv_df["recency"] / 7  # recency değeri haftalık olmalıdır !
    cltv_df["T"] = cltv_df["T"] / 7  # T - müşteri yaşı değeri haftalık olmalıdır !
    # BG - NBD Modeli set ediliyor
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    # penalizer_coef parametrelerin bulunması esnasında paramaetrelere uygulanacak ceza katsayısıdır.
    bgf.fit(cltv_df["frequency"],
            cltv_df["recency"],
            cltv_df["T"])
    cltv_df["expected_purc_1_week"] = bgf.conditional_expected_number_of_purchases_up_to_time(1, cltv_df["frequency"],
                                                                                              cltv_df["recency"],
                                                                                              cltv_df["T"])
    cltv_df["expected_purc_1_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4, cltv_df["frequency"],
                                                                                               cltv_df["recency"],
                                                                                               cltv_df["T"])
    cltv_df["expected_purc_3_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4*3, cltv_df["frequency"],
                                                                                               cltv_df["recency"],
                                                                                               cltv_df["T"])
    # Gamma Gamma Modeli Set Ediliyor
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df["frequency"], cltv_df["monetary"])
    cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                                 cltv_df["monetary"]).sort_values(ascending=False)
    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df["frequency"],
                                       cltv_df["recency"],
                                       cltv_df["T"],
                                       cltv_df["monetary"],
                                       time=month,  # 3 aylık. time ay cinsinden
                                       freq="W",  # T'nin frekans bilgisi
                                       discount_rate=0.01  # the monthly adjusted discount rate
                                       )
    cltv.reset_index()
    cltv_final = cltv_df.merge(cltv, how="left", on="Customer ID")
    cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

    return cltv_final
df = df_.copy()
cltv_final2 = create_cltv_p(df)
cltv_final2.to_excel("cltv_prediction.xlsx")
