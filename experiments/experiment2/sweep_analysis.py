import pandas as pd
from scipy.stats import ttest_ind

df1 = pd.read_csv('Q-Learning-sweep-results.csv')
df2 = pd.read_csv('MC-sweep-results.csv')

df1_gamma_99 = df1[df1['hparams'] == 'gamma:0.99']['last_mean_return']
df1_gamma_08 = df1[df1['hparams'] == 'gamma:0.8']['last_mean_return']
df2_gamma_99 = df2[df2['hparams'] == 'gamma:0.99']['last_mean_return']
df2_gamma_08 = df2[df2['hparams'] == 'gamma:0.8']['last_mean_return']

t_stat1, p_val1 = ttest_ind(df1_gamma_99, df1_gamma_08)
t_stat2, p_val2 = ttest_ind(df2_gamma_99, df2_gamma_08)

print(f"Q-Learning: T-statistic = {t_stat1}, P-value = {p_val1}")
print(f"Monte Carlo: T-statistic = {t_stat2}, P-value = {p_val2}")