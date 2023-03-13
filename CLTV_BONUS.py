import pandas as pd, datetime as dt

df_ = pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.head()
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 800)
pd.set_option("display.float_format", lambda x: "%.5f" % x)



def check_df(dataframe, head=5):
    print("########################### Shape #################")
    print(dataframe.shape)
    print("########################### Types #################")
    print(dataframe.dtypes)  # pandas dataframe de dtype yerine dtypes kullanılmalı!!!
    print("########################### Head #################")
    print(dataframe.head(head))
    print("########################### Tail #################")
    print(dataframe.tail(head))
    print("########################### NA #################")
    print(dataframe.isnull().sum())
    print("########################### Quantiles #################")
    print(dataframe.describe([0, 0.05, 0.25, 0.95, 0.99, 1]).T)
    check_df(df)
    df.info()
    df.head()
    df[df["Price"] < 0]
    df = df[(df["Price"] > 0) & (df["Quantity"] > 0)]
    df.describe().T
    df[~df["Invoice"].str.contains("C", na=False)]
    df.isnull().sum()
    df = df.dropna(subset=["Customer ID"])
    df["Description"].value_counts().shape[0]
    df.groupby("Description").agg({"Quantity": lambda x: x.sum()}).sort_values(by="Quantity", ascending=False).head(20)
    df = df[~df["Invoice"].str.contains("C", na=False)]
    df["TotalPrice"] = df["Price"] * df["Quantity"]
    df.shape
    today_date = df["InvoiceDate"].max() + dt.timedelta(2)  # songünkü rapor tarihine 2 gün ekledik !
    rfm = df.groupby("Customer ID").agg({"InvoiceDate": lambda x: (today_date - x.max()).days, #receency
                                   "Invoice": lambda x: x.nunique(), #frequency
                                   "TotalPrice": "sum"
                                   })
    rfm.columns = ["recency", "frequency", "monetary"]
    rfm["recency_score"] = pd.qcut(rfm["recency"], q=5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(rfm["monetary"], q=5, labels=[1, 2, 3, 4, 5])


    rfm.head()
    rfm["RF_score"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)
    # rfm kural tabanlı müşteri segmentasyon yöntemidir. k-means de ise clustering yapılabilir.
    # gözlem birimlerinin özelliklerinin matematiksek olarak birbirlerine yakınlıklarının belirlenmesidir
    # sınıfladnırma: makine öğrenmesinde var olan sınıflardan hangisine ait olduğunu belirlemek
    # segmente etmek: business hedefine göre gruplara ayırma işdir
    # kümeleme: ortak özelliklere göre, matematiksel özelliklere göre gruplamaktadır
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
    
    rfm["segment"] = rfm["RF_score"].replace(seg_map, regex=True)
    rfm.head()
    rfm.reset_index(inplace=True)
    rfm[["segment","recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "sum", "count"])
    rfm[rfm["segment"] == "loyal_customers"]["Customer ID"].to_excel("loyals.xlsx", index=True)


##################CLTV PREDICTION
df = df_.copy()
df = df[~df["Invoice"].str.contains("C", na=False)]
df.describe().T
today_date = df["InvoiceDate"].max() + dt.timedelta(2)  # songünkü rapor tarihine 2 gün ekledik !
df.dropna(inplace=True)
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = round(quartile3 + 1.5 * interquantile_range)
    low_limit = round(quartile1 - 1.5 * interquantile_range)
    return low_limit, up_limit
# aşağıdaki fonksiyon ile low limit ve up limit üzerindeki değişken değerleri(outlier / aykırı değer) low limit ve
# up limit e set edilerek aykırı değerler baskılanıyor !

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit # veri setini okuttuktan sonra alt limit aykırı dğeerler silineceğinden bunu kapattık
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
replace_with_thresholds(df, "Price")
replace_with_thresholds(df, "Quantity")
replace_with_thresholds(cltv_df, "frequency")


df["TotalPrice"] = df["Price"] * df["Quantity"]
df.describe()
# recency: Son satın alma üzerinden geçen zaman. Haftalık. (kullanıcı özelinde) -  müşterinin kendi içinde son satın alma zamanı - ilk satın alma zamanını ifade eder
# T: Müşterinin yaşı. Haftalık. (analiz tarihinden ne kadar süre önce ilk satın alma yapılmış)
# frequency: tekrar eden toplam satın alma sayısı (frequency>1)
# monetary: satın alma başına ortalama kazanç
cltv_df = df.groupby("Customer ID").agg({"InvoiceDate": [lambda x: (x.max() - x.min()).days, #recency
                                                         lambda x:(today_date - x.min()).days ],#Tenure
                                         "Quantity": "sum", #frequency
                                         "TotalPrice": "sum"})
df.describe().T
cltv_df.columns = cltv_df.columns.droplevel(1)
cltv_df.columns = ["recency", "T", "frequency", "monetary"]
cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7
cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

cltv_df = cltv_df[(cltv_df["monetary"] > 0) & (cltv_df["frequency"] > 1)]
cltv_df = cltv_df[cltv_df["recency"]>0]
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
replace_with_thresholds(cltv_df, "frequency")
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df["frequency"],
            cltv_df["recency"],
            cltv_df["T"])
cltv_df.describe().T
bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        cltv_df["frequency"],
                                                        cltv_df["recency"],
                                                        cltv_df["T"])
cltv_df.head()
cltv_df
### gamma gamma submodel kurulması
ggf = GammaGammaFitter(penalizer_coef=0.1)
ggf.fit(cltv_df["frequency"],
        cltv_df["monetary"])
cltv_df.reset_index()
cltv_df = cltv_df[cltv_df["frequency"] > 1]
cltv_df = cltv_df[cltv_df["monetary"] > 0]
conditue = ggf.conditional_expected_average_profit(cltv_df["frequency"],cltv_df["monetary"])
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df["frequency"],
                                   cltv_df["recency"],
                                   cltv_df["T"],
                                   cltv_df["monetary"],
                                   time=3, # 3 aylık. time ay cinsinden
                                   freq="W", #T'nin frekans bilgisi
                                   discount_rate=0.01 )# the monthly adjusted discount rate

cltv_df.merge(cltv, how="left", on="Customer ID")
conditue= pd.DataFrame(conditue)
cltv_df.merge(conditue, how="left", on="Customer ID")
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary', "exp_avg"]