import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

transactions = pd.read_parquet('transaction_fraud_data.parquet')
currency = pd.read_parquet('historical_currency_exchange.parquet')

print(transactions.info())
print(currency.info())


fraud_ratio = transactions['is_fraud'].value_counts(normalize=True)
print(f"Доля мошеннических транзакций: {fraud_ratio[1]*100:.2f}%")


plt.figure(figsize=(8, 6))
sns.countplot(x='is_fraud', data=transactions)
plt.title('Распределение мошеннических транзакций')
plt.show()


#Топ категорий по мошенничеству
fraud_by_category = transactions.groupby('vendor_category')['is_fraud'].mean().sort_values(ascending=False)
print(fraud_by_category)

plt.figure(figsize=(12, 6))
fraud_by_category.plot(kind='bar')
plt.title('Доля мошенничества по категориям вендоров')
plt.ylabel('Доля мошенничества')
plt.show()

#Топ стран по мошенничеству
fraud_by_country = transactions.groupby('country')['is_fraud'].mean().sort_values(ascending=False).head(10)
print(fraud_by_country)

plt.figure(figsize=(12, 6))
fraud_by_country.plot(kind='bar')
plt.title('Топ-10 стран по доле мошеннических транзакций')
plt.ylabel('Доля мошенничества')
plt.show()

#Распределение по типам карт
fraud_by_card = transactions.groupby('card_type')['is_fraud'].mean().sort_values(ascending=False)
print(fraud_by_card)

plt.figure(figsize=(12, 6))
fraud_by_card.plot(kind='bar')
plt.title('Доля мошенничества по типам карт')
plt.ylabel('Доля мошенничества')
plt.show()

#Распределение сумм
plt.figure(figsize=(12, 6))
sns.boxplot(x='is_fraud', y='amount', data=transactions)
plt.yscale('log') 
plt.title('Распределение сумм транзакций по признаку мошенничества')
plt.show()

transactions['hour'] = transactions['timestamp'].dt.hour

#Мошенничество по часам
fraud_by_hour = transactions.groupby('hour')['is_fraud'].mean()

plt.figure(figsize=(12, 6))
fraud_by_hour.plot()
plt.title('Доля мошенничества по часам дня')
plt.ylabel('Доля мошенничества')
plt.xlabel('Час дня')
plt.show()