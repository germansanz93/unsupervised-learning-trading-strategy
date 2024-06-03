from statsmodels.regression.rolling import RollingOLS
import pandas_datareader as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import pandas_ta
import warnings
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import matplotlib.ticker as mticker

## imports de la segunda parte, la de ML
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')

ATR_COL = 5
RSI_COL = 1


def compute_atr(stock_data):
    atr = pandas_ta.atr(high=stock_data['high'], low=stock_data['low'], close=stock_data['close'], length=14)
    return atr.sub(atr.mean()).div(atr.std())


def compute_macd(close):
    macd = pandas_ta.macd(close=close, length=20).iloc[:, 0]
    return macd.sub(macd.mean()).div(macd.std())


def calculate_returns(dfa):
    outlier_cutoff = 0.005

    lags = [1, 2, 3, 6, 9, 12]  # Lags en meses, 6 diferentes

    for lag in lags:
        dfa[f'return_{lag}m'] = (dfa['adj close'].pct_change(lag).pipe(
            lambda x: x.clip(lower=x.quantile(outlier_cutoff), upper=x.quantile(1 - outlier_cutoff))).add(1).pow(
            1 / lag).sub(1))

    return dfa


# Funciones de la segunda parte, ML
def get_clusters(df):
    df['cluster'] = KMeans(n_clusters=4,
                           random_state=0,
                           init=get_initial_centroids()).fit(df).labels_
    return df


def get_initial_centroids():
    target_rsi_values = [30, 45, 55, 70]

    initial_centroids = np.zeros((len(target_rsi_values), 18))
    initial_centroids[:, RSI_COL] = target_rsi_values
    return initial_centroids


def plot_clusters(dataframe):
    cluster_0 = dataframe[dataframe['cluster'] == 0]
    cluster_1 = dataframe[dataframe['cluster'] == 1]
    cluster_2 = dataframe[dataframe['cluster'] == 2]
    cluster_3 = dataframe[dataframe['cluster'] == 3]

    # plt.scatter(cluster_0.loc[:, 'atr'], cluster_0.loc[:, 'rsi'], color='red', label='cluster 0') #En lugar de loc y el nombre de la col, podria usar iloc y el indice de dicha col y seria muchisimo mas rapido.
    plt.scatter(cluster_0.iloc[:, ATR_COL], cluster_0.iloc[:, RSI_COL], color='red', label='cluster 0')
    plt.scatter(cluster_1.iloc[:, ATR_COL], cluster_1.iloc[:, RSI_COL], color='green', label='cluster 1')
    plt.scatter(cluster_2.iloc[:, ATR_COL], cluster_2.iloc[:, RSI_COL], color='blue', label='cluster 2')
    plt.scatter(cluster_3.iloc[:, ATR_COL], cluster_3.iloc[:, RSI_COL], color='black', label='cluster 3')

    plt.legend()
    plt.show()
    return


# portfolio optimization
def optimize_weights(prices, lower_bound=0.0):
    returns = expected_returns.mean_historical_return(prices=prices,
                                                      frequency=252)

    cov = risk_models.sample_cov(prices=prices,
                                 frequency=252)

    ef = EfficientFrontier(expected_returns=returns,
                           cov_matrix=cov,
                           weight_bounds=(lower_bound, 1),
                           # TODO este limite lo pone en 0.1... por que??? no deberia ser 1 para que el limite superior sea 100%?
                           solver='SCS')

    weights = ef.max_sharpe()

    return ef.clean_weights()


if __name__ == '__main__':
    YEARS = 8
    sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')
    symbols_list = sp500['Symbol'].unique().tolist()
    end_date = '2024-04-01'
    start_date = pd.to_datetime(end_date) - pd.DateOffset(365 * YEARS)

    df = yf.download(tickers=symbols_list, start=start_date, end=end_date).stack()

    df.index.names = ['date', 'ticker']

    df.columns = df.columns.str.lower()

    df['garman_klass_vol'] = ((np.log(df['high']) - np.log(df['low'])) ** 2) / 2 - (2 * np.log(2) - 1) * (
            (np.log(df['adj close']) - np.log(df['open'])) ** 2)

    df['rsi'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.rsi(close=x, length=20))

    # df.xs('AAPL', level=1)['rsi'].plot()
    # pandas_ta.bbands(close=df.xs('AAPL', level=1)['adj close'], length=20)

    df['bb_low'] = df.groupby(level=1)['adj close'].transform(
        lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:, 0])
    df['bb_mid'] = df.groupby(level=1)['adj close'].transform(
        lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:, 1])
    df['bb_high'] = df.groupby(level=1)['adj close'].transform(
        lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:, 2])

    df['atr'] = df.groupby(level=1, group_keys=False).apply(compute_atr)

    df['macd'] = df.groupby(level=1, group_keys=False)['adj close'].apply(compute_macd)

    df['dollar_volume'] = (df['adj close'] * df['volume']) / 1e6

    last_cols = [c for c in df.columns.unique(0) if
                 c not in ['dollar_volume', 'volume', 'open', 'high', 'low', 'close']]
    # print(last_cols)
    data = (
        pd.concat([df.unstack('ticker')['dollar_volume'].resample('M').mean().stack('ticker').to_frame('dollar_volume'),
                   df.unstack()[last_cols].resample('M').last().stack('ticker')], axis=1)).dropna()

    data['dollar_volume'] = data.loc[:, 'dollar_volume'].unstack('ticker').rolling(5 * 12,
                                                                                   min_periods=12).mean().stack()

    data['dollar_vol_rank'] = data.groupby('date')['dollar_volume'].rank(ascending=False)

    data = data[data['dollar_vol_rank'] < 150].drop(['dollar_volume', 'dollar_vol_rank'],
                                                    axis=1)  # Las 150 acciones con mas volumen

    g = df.xs('AAPL', level=1)

    data = data.groupby(level=1, group_keys=False).apply(calculate_returns).dropna()
    factor_data = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start='2010')[0].drop('RF', axis=1)
    factor_data.index = pd.to_datetime(factor_data.index.to_timestamp())
    factor_data = factor_data.resample('M').last()
    factor_data.index.name = 'date'

    factor_data = factor_data.join(data['return_1m']).sort_index()

    # print(factor_data.xs('AAPL', level=1).head())
    # print(factor_data.xs('GOOGL', level=1).head())

    observations = factor_data.groupby(level=1).size()  # Cantidad de meses de data por accion

    valid_stocks = observations[observations >= 10]

    factor_data = factor_data[factor_data.index.get_level_values('ticker').isin(valid_stocks.index)]

    betas = (factor_data.groupby(level=1, group_keys=False)
             .apply(lambda x: RollingOLS(endog=x['return_1m'],
                                         exog=sm.add_constant(x.drop('return_1m', axis=1)),
                                         window=min(24, x.shape[0]),
                                         min_nobs=len(x.columns) + 1).fit(params_only=True).params.drop('const',
                                                                                                        axis=1)))
    data = data.join(betas.groupby('ticker').shift())

    factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']

    data.loc[:, factors] = data.groupby('ticker', group_keys=False)[factors].apply(lambda x: x.fillna(x.mean()))

    data = data.dropna()
    data = data.drop('adj close', axis=1)

    # print(data.info())

    # print(factor_data)
    # print(data)

    ## Ahora que tenemos la data limpia, ordenada y normalizada viene la aplicacion de ML a too esto

    data = data.groupby('date', group_keys=False).apply(get_clusters)

    plt.style.use('ggplot')

    # esta funcion deberia hacer los graficos, pero andasa como se hace que se generen
    for i in data.index.get_level_values('date').unique().tolist():
        g = data.xs(i, level=0)
        plt.title(f'Date {i}')
        plot_clusters(g)
        plt.savefig(f'grafico_{i}.png')
        plt.close()

    filtered_df = data[data['cluster'] == 3].copy()

    filtered_df = filtered_df.reset_index(level=1)

    filtered_df.index = filtered_df.index + pd.DateOffset(1)

    filtered_df = filtered_df.reset_index().set_index(['date', 'ticker'])

    dates = filtered_df.index.get_level_values('date').unique().tolist()

    fixed_dates = {}

    for d in dates:
        fixed_dates[d.strftime('%Y-%m-%d')] = filtered_df.xs(d, level=0).index.tolist()

    # optimization
    stocks = data.index.get_level_values('ticker').unique().tolist()

    start = data.index.get_level_values('date').unique()[0] - pd.DateOffset(months=12)

    new_df = yf.download(tickers=stocks,
                         start=data.index.get_level_values('date').unique()[0] - pd.DateOffset(months=12),
                         end=data.index.get_level_values('date').unique()[-1])

    returns_dataframe = np.log(new_df['Adj Close']).diff()

    portfolio_df = pd.DataFrame()

    for start_date in fixed_dates.keys():
        try:
            end_date = (pd.to_datetime(start_date) + pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')

            cols = fixed_dates[start_date]

            optimization_start_date = (pd.to_datetime(start_date) - pd.DateOffset(months=12)).strftime('%Y-%m-%d')

            optimization_end_date = (pd.to_datetime(start_date) - pd.DateOffset(days=1)).strftime('%Y-%m-%d')

            optimization_df = new_df[optimization_start_date:optimization_end_date]['Adj Close'][cols]

            success = False
            try:

                weights = optimize_weights(prices=optimization_df,
                                           lower_bound=round(1 / (len(optimization_df.columns) * 2), 3))

                weights = pd.DataFrame(weights, index=pd.Series(0))

                success = True

            except:
                print(f'Max Sharpe Optimization failed for {start_date}, Continuig with Equal-Weights')

            if not success:
                weights = pd.DataFrame([1 / len(optimization_df.columns) for i in range(len(optimization_df.columns))],
                                       index=optimization_df.columns.tolist(),
                                       columns=pd.Series(0)).T
                # print(weights)

            temp_df = returns_dataframe[start_date:end_date]
            temp_df = temp_df.stack().to_frame('return').reset_index(level=0) \
                .merge(weights.stack().to_frame('weight').reset_index(level=0, drop=True),
                       left_index=True,
                       right_index=True) \
                .reset_index().set_index(['Date', 'Ticker']).unstack().stack()

            temp_df.index.names = ['date', 'ticker']

            temp_df['weighted_return'] = temp_df['return'] * temp_df['weight']

            temp_df = temp_df.groupby(level=0)['weighted_return'].sum().to_frame('Strategy Return')

            portfolio_df = pd.concat([portfolio_df, temp_df], axis=0)

        except Exception as e:
            print(e)

    portfolio_df.plot()
    plt.savefig('strategy.png')

    portfolio_df = portfolio_df.drop_duplicates()

    spy = yf.download(tickers='SPY', start='2018-04-30', end=dt.date.today())
    spy_ret = np.log(spy[['Adj Close']]).diff().dropna().rename({'Adj Close':'SPY Buy&Hold'}, axis=1)
    portfolio_df = portfolio_df.merge(spy_ret, left_index=True, right_index=True)

    print(portfolio_df)

    plt.style.use('ggplot')

    portfolio_cumulative_return = np.exp(np.log1p(portfolio_df).cumsum()) - 1

    portfolio_cumulative_return[:'2024-03-28'].plot(figsize=(16, 6))

    plt.title('Unsupervised Learning Trading Strategy Returns Over Time')

    plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(1))

    plt.ylabel("Return")

    plt.savefig('strategies-comparisson.png')