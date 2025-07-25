# %%
import os
import pandas as pd
import streamlit as st
# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# %%


def columns_datatype_change(df):
    for col in df.columns:
        date_change = df[col].astype(str).str.match(
            r'^\d{4}/\d{1,2}/\d{1,2}$', na=False).any()
        if date_change:
            df[col] = pd.to_datetime(
                df[col], format='%Y/%m/%d', errors='coerce')

    return df


font_prop = fm.FontProperties(fname='fonts/ipaexg.ttf')
# 棒グラフ×折れ線グラフ


def bar_line_plot(df, x_columun, y_columun1, y_columun2, title):
    # グラフの描画設定
    fig = plt.figure(figsize=(7, 4), facecolor="w")
    plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams["font.size"] = 10
    # X軸を非連続にするために文字列に変換
    df['x_numeric'] = df[x_columun].astype(str)
    ax1 = fig.subplots()
    ax2 = ax1.twinx()
    # Y軸の最小、最大値を設定
    ax1.set_ylim([0, df[y_columun1].max()*1.1])
    ax2.set_ylim([0, df[y_columun2].max()*1.1])
    # Y軸目盛の描画を設定
    ax1.grid(True)
    ax2.grid(False)
    # グラフ色を設定
    color1 = "darkblue"
    color2 = "orange"
    # グラフ描画
    sns.barplot(x='x_numeric', y=y_columun1, data=df,
                ax=ax1, color=color1, label='売上[USドル]')
    sns.lineplot(x='x_numeric', y=y_columun2, data=df,
                 ax=ax2, color=color2, label="利益率[%]")
    # 凡例を取得
    handler1, label1 = ax1.get_legend_handles_labels()
    handler2, label2 = ax2.get_legend_handles_labels()
    # ax1で凡例をまとめて表示
    ax1.legend(handler1 + handler2, label1 + label2,
               loc="lower left", fontsize=10, prop=font_prop)
    # ax2の凡例は削除
    ax2.get_legend().remove()
    # タイトル、軸ラベルの設定
    plt.title(title, fontsize=15, fontproperties=font_prop)
    ax1.set_xlabel('', fontproperties=font_prop)
    ax1.set_ylabel('売上[USドル]', fontsize=15, fontproperties=font_prop)
    ax2.set_ylabel('利益率[%]', fontsize=15, fontproperties=font_prop)
    ax1.set_xticklabels(ax1.get_xticklabels(), fontproperties=font_prop)
    plt.tight_layout()
    return fig

# 複数棒グラフ×折れ線グラフ


def stakedbar_line_plot(df, x_columun, y_columun1, y_columun2, stacked_columun1, title):
    # グラフの描画設定
    fig = plt.figure(figsize=(7, 4), facecolor="w")
    plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams["font.size"] = 10
    # X軸を非連続にするために文字列に変換
    df['x_numeric'] = df[x_columun].astype(str)
    ax1 = fig.subplots()
    ax2 = ax1.twinx()
    # Y軸の最小、最大値を設定
    ax1.set_ylim([0, df[y_columun1].max()*1.1])
    ax2.set_ylim([0, df[y_columun2].max()*1.1])
    # Y軸目盛の描画を設定
    ax1.grid(True)
    ax2.grid(False)
    # グラフ色を設定
    palette1 = 'pastel'
    palette2 = 'hls'
    # グラフ描画
    sns.barplot(x='x_numeric', y=y_columun1, data=df, ax=ax1,
                hue=stacked_columun1, palette=palette1)
    sns.lineplot(x='x_numeric', y=y_columun2, data=df, ax=ax2,
                 hue=stacked_columun1, palette=palette2)
    # 凡例の位置とフォントサイズを変更
    ax1.legend(loc="lower left", fontsize=10, prop=font_prop)
    ax2.legend(loc="lower right", fontsize=10, prop=font_prop)
    # タイトル、軸ラベルの設定
    plt.title(title, fontsize=15, fontproperties=font_prop)
    ax1.set_xlabel('', fontproperties=font_prop)
    ax1.set_ylabel('売上[USドル]', fontsize=15, fontproperties=font_prop)
    ax2.set_ylabel('利益率[%]', fontsize=15, fontproperties=font_prop)
    ax1.set_xticklabels(ax1.get_xticklabels(), fontproperties=font_prop)
    plt.tight_layout()
    return fig

# 棒グラフ


def bar_plot(df, x_columun, y_columun, y_rabel, title, rotation):
    # グラフの描画設定
    fig, ax = plt.subplots(figsize=(8, 3), facecolor="w")
    plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams["font.size"] = 10
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    # X軸を非連続にするために文字列に変換
    df['x_numeric'] = df[x_columun].astype(str)
    sns.barplot(x='x_numeric', y=y_columun, data=df, ax=ax, color='darkblue')
    plt.xticks(rotation=rotation)
    # タイトル、軸ラベルの設定
    plt.title(title, fontsize=15, fontproperties=font_prop)
    ax.set_xlabel('', fontproperties=font_prop)
    ax.set_ylim([0, df[y_columun].max()*1.1])
    ax.set_ylabel(y_rabel, fontsize=10, fontproperties=font_prop)
    ax.set_xticklabels(ax.get_xticklabels(), fontproperties=font_prop)
    plt.tight_layout()
    return fig


# 複数棒グラフ


def stakedbar_plot(df, x_columun, y_columun, y_rabel, stacked_columun, title, rotation):
    # グラフの描画設定
    fig, ax = plt.subplots(figsize=(7, 4), facecolor="w")
    plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams["font.size"] = 10
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    # X軸を非連続にするために文字列に変換
    df['x_numeric'] = df[x_columun].astype(str)
    pivot_df = df.pivot_table(index=x_columun, columns=stacked_columun,
                              values=y_columun, aggfunc='sum', fill_value=0)
    pivot_df.plot(kind='bar', stacked=True, ax=ax,
                  color=['darkblue', 'orange'])
    plt.xticks(rotation=rotation)
    # 凡例の位置とフォントサイズを変更
    ax.legend(loc="lower left", fontsize=10, prop=font_prop)
    # タイトル、軸ラベルの設定
    plt.title(title, fontsize=15, fontproperties=font_prop)
    ax.set_xlabel('', fontproperties=font_prop)
    ax.set_ylabel(y_rabel, fontsize=15, fontproperties=font_prop)
    ax.set_xticklabels(ax.get_xticklabels(), fontproperties=font_prop)
    plt.tight_layout()
    return fig


# 散布図


def catplot_strip(df, x_columun, y_columun, y_rabel, title):
    # グラフの描画設定
    fig, ax = plt.subplots(figsize=(8, 3), facecolor="w")
    plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams["font.size"] = 10
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    # X軸を非連続にするために文字列に変換
    df['x_numeric'] = df[x_columun].astype(str)
    sns.stripplot(x='x_numeric', y=y_columun, data=df, ax=ax, color='darkblue')
    plt.xticks(rotation=90)
    # タイトル、軸ラベルの設定
    plt.title(title, fontsize=15, fontproperties=font_prop)
    ax.set_xlabel('', fontproperties=font_prop)
    ax.set_ylabel(y_rabel, fontsize=10, fontproperties=font_prop)
    ax.axhline(y=0, color='gray', linestyle='--')
    ax.set_xticklabels(ax.get_xticklabels(), fontproperties=font_prop)
    plt.tight_layout()
    return fig
# 折れ線グラフ


def line_plot(df, x_columun, y_columun, y_rabel, title):
    # グラフの描画設定
    fig, ax = plt.subplots(figsize=(8, 2), facecolor="w")
    plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams["font.size"] = 10
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    # グラフ描画
    sns.lineplot(x=x_columun, y=y_columun, data=df, ax=ax, color='darkblue')
    # タイトル、軸ラベルの設定
    plt.title(title, fontsize=15, fontproperties=font_prop)
    ax.set_xlabel('')
    if title == 'リピート率の推移※2014は初年度のため無し':
        plt.xticks(df['year'])
    ax.set_ylabel(y_rabel, fontsize=10, fontproperties=font_prop)
    plt.tight_layout()
    return fig


def reg_plot(df, x_columun, y_columun, x_rabel, y_rabel, title):
    # グラフの描画設定
    fig, ax = plt.subplots(figsize=(4, 2), facecolor="w")
    plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams["font.size"] = 10
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    # グラフ描画
    sns.regplot(x=x_columun, y=y_columun, data=df, ax=ax, scatter_kws={
                "color": "darkblue"}, line_kws={"color": "orange"})
    # タイトル、軸ラベルの設定
    plt.title(title, fontsize=15, fontproperties=font_prop)
    ax.set_xlabel(x_rabel, fontsize=10, fontproperties=font_prop)
    ax.set_ylabel(y_rabel, fontsize=10, fontproperties=font_prop)
    ax.set_xticklabels(ax.get_xticklabels(), fontproperties=font_prop)
    plt.tight_layout()
    return fig
# plotly版


def bar_line_plot_plotly(df, x_columun, y_columun1, y_columun2, title):
    df['x_numeric'] = df[x_columun].astype(str)

    fig = go.Figure()

    # 棒グラフ（売上）
    fig.add_trace(go.Bar(
        x=df['x_numeric'],
        y=df[y_columun1],
        name='売上[USドル]',
        marker_color='darkblue',
        yaxis='y1'
    ))

    # 折れ線グラフ（利益率）
    fig.add_trace(go.Scatter(
        x=df['x_numeric'],
        y=df[y_columun2],
        name='利益率[%]',
        mode='lines+markers',
        line=dict(color='orange'),
        yaxis='y2'
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(title=''),
        yaxis=dict(title='売上[USドル]', side='left',
                   range=[0, df[y_columun1].max()*1.1]),
        yaxis2=dict(title='利益率[%]', overlaying='y', side='right', range=[
                    0, df[y_columun2].max()*1.1]),
        legend=dict(x=0, y=0, xanchor='left', yanchor='bottom'),
        margin=dict(l=40, r=40, t=40, b=40),
        template='simple_white',
        font=dict(family='IPAexGothic', size=12),
        height=300
    )
    return fig


def stakedbar_line_plot_plotly(df, x_columun, y_columun1, y_columun2, stacked_columun1, title):
    df['x_numeric'] = df[x_columun].astype(str)

    fig = go.Figure()

    # 積み上げ棒グラフ（売上）
    for key in df[stacked_columun1].unique():
        subset = df[df[stacked_columun1] == key]
        fig.add_trace(go.Bar(
            x=subset['x_numeric'],
            y=subset[y_columun1],
            name=f'{key}',
            yaxis='y1'
        ))

    # 折れ線グラフ（利益率）
    for key in df[stacked_columun1].unique():
        subset = df[df[stacked_columun1] == key]
        fig.add_trace(go.Scatter(
            x=subset['x_numeric'],
            y=subset[y_columun2],
            mode='lines+markers',
            name=f'{key}（利益率）',
            yaxis='y2'
        ))

    fig.update_layout(
        title=title,
        barmode='group',
        xaxis=dict(title=''),
        yaxis=dict(title='売上[USドル]', side='left',
                   range=[0, df[y_columun1].max()*1.1]),
        yaxis2=dict(title='利益率[%]', overlaying='y', side='right', range=[
                    0, df[y_columun2].max()*1.1]),
        legend=dict(x=0, y=0, xanchor='left',
                    yanchor='bottom', font=dict(size=8)),
        margin=dict(l=40, r=40, t=40, b=40),
        template='simple_white',
        font=dict(family='IPAexGothic', size=12),
        height=600
    )
    return fig


def bar_plot_plotly(df, x_columun, y_columun, y_rabel, title, rotation):
    df['x_numeric'] = df[x_columun].astype(str)

    fig = go.Figure(data=[
        go.Bar(
            x=df['x_numeric'],
            y=df[y_columun],
            marker_color='darkblue'
        )
    ])

    fig.update_layout(
        title=title,
        xaxis=dict(title='', tickangle=rotation, tickfont=dict(size=8)),
        yaxis=dict(title=y_rabel, range=[0, df[y_columun].max()*1.1]),
        template='plotly',
        font=dict(family='IPAexGothic', size=12),
        height=300
    )
    return fig


def stakedbar_plot_plotly(df, x_columun, y_columun, y_rabel, stacked_columun, title, rotation):
    df['x_numeric'] = df[x_columun].astype(str)

    pivot_df = df.pivot_table(index='x_numeric', columns=stacked_columun,
                              values=y_columun, aggfunc='sum', fill_value=0)
    pivot_df = pivot_df.reset_index()

    fig = go.Figure()
    for col in pivot_df.columns[1:]:
        fig.add_trace(go.Bar(
            x=pivot_df['x_numeric'],
            y=pivot_df[col],
            name=col
        ))

    fig.update_layout(
        title=title,
        barmode='stack',
        xaxis=dict(title='', tickangle=rotation),
        yaxis=dict(title=y_rabel),
        template='simple_white',
        font=dict(family='IPAexGothic', size=12),
        height=300
    )
    return fig


def catplot_strip_plotly(df, x_columun, y_columun, y_rabel, title):
    df['x_numeric'] = df[x_columun].astype(str)

    fig = go.Figure(data=[
        go.Scatter(
            x=df['x_numeric'],
            y=df[y_columun],
            mode='markers',
            marker=dict(color='darkblue'),
        )
    ])

    fig.add_hline(y=0, line_dash="dash", line_color="orange")

    fig.update_layout(
        title=title,
        xaxis=dict(title='', tickangle=90, tickfont=dict(size=8)),
        yaxis=dict(title=y_rabel),
        template='simple_white',
        font=dict(family='IPAexGothic', size=12),
        height=300
    )
    return fig


def line_plot_plotly(df, x_columun, y_columun, y_rabel, title):
    fig = go.Figure(data=[
        go.Scatter(
            x=df[x_columun],
            y=df[y_columun],
            mode='lines+markers',
            marker_color='darkblue'
        )
    ])

    fig.update_layout(
        title=title,
        xaxis=dict(title='', tickformat="%m/%d", dtick="D5", tickangle=90),
        yaxis=dict(title=y_rabel),
        template='simple_white',
        font=dict(family='IPAexGothic', size=12),
        height=300
    )
    return fig


def reg_plot_plotly(df, x_columun, y_columun, x_rabel, y_rabel, title):
    # 回帰線の係数を取得
    coeffs = np.polyfit(df[x_columun], df[y_columun], 1)
    regression_line = np.poly1d(coeffs)(df[x_columun])

    fig = go.Figure()

    # 散布図
    fig.add_trace(go.Scatter(
        x=df[x_columun],
        y=df[y_columun],
        mode='markers',
        marker=dict(color='darkblue'),
        name='データ'
    ))

    # 回帰線
    fig.add_trace(go.Scatter(
        x=df[x_columun],
        y=regression_line,
        mode='lines',
        line=dict(color='orange'),
        name='回帰線'
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(title=x_rabel),
        yaxis=dict(title=y_rabel),
        template='simple_white',
        font=dict(family='IPAexGothic', size=12),
        height=300
    )
    return fig


# %% [markdown]
#

# %%
# 元データ読み込み
orders_df = pd.read_csv('data_csv/Orders.csv')
# people_df= pd.read_csv('data_csv/People.csv')
returns_df = pd.read_csv('data_csv/Returns.csv')
language_df = pd.read_csv('data_csv/language.csv')
# データ型の調査
print(orders_df.dtypes)
# print(people_df.dtypes)
print(returns_df.dtypes)
print(language_df.dtypes)
# Null有無の調査
print(orders_df.isnull().values.sum())
# print(people_df.isnull().values.sum())
print(returns_df.isnull().values.sum())

# カラム文字列の判別結果から日付け型に変換
orders_df = columns_datatype_change(orders_df)

# 使用するテーブルを結合
merge_df = pd.merge(orders_df, returns_df, how='left', on='Order ID')

# %%

merge_df['year'] = merge_df['Order Date'].dt.year.astype(int)
merge_df['month'] = merge_df['Order Date'].dt.month
merge_df = merge_df.replace(
    {'Region': {'South': '南', 'Central': '中央', 'East': '東', 'West': '西'}})
merge_df = merge_df.replace({'State':
                             {"California": "カリフォルニア"
                              , "Florida": "フロリダ"
                              , "North Carolina": "ノースカロライナ"
                              , "Washington": "ワシントン"
                              , "Texas": "テキサス"
                              , "Wisconsin": "ウィスコンシン"
                              , "Utah": "ユタ"
                              , "Nebraska": "ネブラスカ"
                              , "Pennsylvania": "ペンシルベニア"
                              , "Illinois": "イリノイ"
                              , "Minnesota": "ミネソタ"
                              , "Michigan": "ミシガン"
                              , "Delaware": "デラウェア"
                              , "Indiana": "インディアナ"
                              , "New York": "ニューヨーク"
                              , "Arizona": "アリゾナ"
                              , "Virginia": "バージニア"
                              , "Tennessee": "テネシー"
                              , "Alabama": "アラバマ"
                              , "South Carolina": "サウスカロライナ"
                              , "Oregon": "オレゴン"
                              , "Colorado": "コロラド"
                              , "Iowa": "アイオワ"
                              , "Ohio": "オハイオ"
                              , "Missouri": "ミズーリ",
                              "Oklahoma": "オクラホマ"
                              , "New Mexico": "ニューメキシコ"
                              , "Louisiana": "ルイジアナ"
                              , "Connecticut": "コネチカット"
                              , "New Jersey": "ニュージャージー"
                              , "Massachusetts": "マサチューセッツ"
                              , "Georgia": "ジョージア"
                              , "Nevada": "ネバダ"
                              , "Rhode Island": "ロードアイランド"
                              , "Mississippi": "ミシシッピ"
                              , "Arkansas": "アーカンソー"
                              , "Montana": "モンタナ"
                              , "New Hampshire": "ニューハンプシャー"
                              , "Maryland": "メリーランド"
                              , "District of Columbia": "ワシントンD.C."
                              , "Kansas": "カンザス"
                              , "Vermont": "バーモント"
                              , "Maine": "メイン"
                              , "South Dakota": "サウスダコタ"
                              , "Idaho": "アイダホ"
                              , "North Dakota": "ノースダコタ"
                              , "Wyoming": "ワイオミング"
                              , "West Virginia": "ウェストバージニア"
                              , "Kentucky": "ケンタッキー"
                              }})
merge_df = merge_df.replace(
    {'Category': {"Furniture": "家具", "Office Supplies": "事務用品", "Technology": "電子機器"}})
merge_df = merge_df.replace({'Sub-Category':
                              {'Bookcases': '本棚'
                               , 'Chairs': '椅子'
                               , 'Labels': 'ラベル'
                               , 'Tables': 'テーブル'
                               , 'Storage': '収納'
                               , 'Furnishings': 'インテリア'
                               , 'Art': 'アート'
                               , 'Phones': '電話'
                               ,'Binders': 'バインダー'
                               , 'Paper': '紙'
                               , 'Accessories': 'アクセサリー'
                               , 'Envelopes': '封筒'
                               , 'Fasteners': '留め具'
                               , 'Supplies': '備品'
                               , 'Appliances': '家電'
                               , 'Copiers': 'コピー機'
                               , 'Machines': '機械'
                               }})
merge_df['Discount_flg'] = merge_df['Discount'].apply(
    lambda x: 1 if x > 0 else 0)
merge_df['Sub-_Category_flg'] = 1
merge_df['Discount'] = merge_df['Discount']*100
merge_df['profit_ratio'] = (merge_df['Profit']/merge_df['Sales'])*100  # 利益率計算
first_order_date = merge_df.groupby(
    'Customer ID')['Order Date'].transform('min')
merge_df['new_customers_flg'] = (
    merge_df['Order Date'] == first_order_date).astype(int)
merge_df['customers_new_old'] = merge_df['new_customers_flg'].apply(
    lambda x: '新規' if x == 1 else '既存')

# %%
# seabornグラフ描画-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# カラム名変数の定義
year_region = ['year', 'Region']
year_month = ['year', 'month']
year_cat = ['year', 'Category']
sls_pr = ['Sales', 'Profit']
# 2016-17売上/利益率
All_sls_pr_fig = bar_line_plot(merge_df.groupby('year', as_index=False)[sls_pr].sum()
                               .assign(profit_ratio=lambda df: (df['Profit'] / df['Sales']) * 100)
                               .pipe(lambda df: df[(df['year'] == 2016) | (df['year'] == 2017)]), 'year', 'Sales', 'profit_ratio', '2016-17売上/利益率')
# #2016-17地域別売上/利益率
reg_sls_pr_fig = stakedbar_line_plot(merge_df.groupby(year_region, as_index=False)[sls_pr].sum()
                                     .assign(profit_ratio=lambda df: (df['Profit'] / df['Sales']) * 100)
                                     .pipe(lambda df: df[(df['year'] == 2016) | (df['year'] == 2017)]), 'year', 'Sales', 'profit_ratio', 'Region', '2016-17地域別売上/利益率')
# 2017南地域州別売上
south_sls_fig = bar_plot(merge_df.groupby(year_region + ['State'], as_index=False)[['Sales']].sum()
                         .pipe(lambda df: df[(df['year'] == 2017) & (df['Region'] == '南')])
                         .sort_values('Sales', ascending=False), 'State', 'Sales', '売上[USドル]', '2017南地域州別売上', 90)
# 2017中央・南地域州別利益率
south_central_pr_fig = catplot_strip(merge_df.groupby(year_region + ['State'], as_index=False)[sls_pr].sum()
                                     .assign(profit_ratio=lambda df: (df['Profit'] / df['Sales']) * 100)
                                     .pipe(lambda df: df[(df['year'] == 2017) & ((df['Region'] == '南') | (df['Region'] == '中央'))])
                                     .sort_values('profit_ratio', ascending=False), 'State', 'profit_ratio', '利益率[%]', '2017中央・南地域州別利益率')
# 2017月別売上
mo_sls_fig = bar_plot(merge_df.groupby(year_month, as_index=False)[sls_pr].sum()
                      .assign(profit_ratio=lambda df: (df['Profit'] / df['Sales']) * 100)
                      .pipe(lambda df: df[df['year'] == 2017]), 'month', 'Sales', '売上[USドル]', '2017月別売上', 0)
# 2017年2月売上
daily_sls_fig = line_plot(merge_df.groupby(['Order Date'] + year_month, as_index=False)[sls_pr].sum()
                          .assign(profit_ratio=lambda df: (df['Profit'] / df['Sales']) * 100)
                          .pipe(lambda df: df[(df['year'] == 2017) & (df['month'] == 2)]), 'Order Date', 'Sales', '売上[USドル]', '2017年2月売上')
# 2016-17カテゴリ別売上/利益率
cat_sls_pr_fig = stakedbar_line_plot(merge_df.groupby(year_cat, as_index=False)[sls_pr].sum()
                                     .assign(profit_ratio=lambda df: (df['Profit'] / df['Sales']) * 100)
                                     .pipe(lambda df: df[(df['year'] == 2016) | (df['year'] == 2017)]), 'year', 'Sales', 'profit_ratio', 'Category', '2016-17カテゴリ別売上/利益率')
# 2017家具/サブカテゴリ別売上
subcat_sls_fig = bar_plot(merge_df.groupby(year_cat + ['Sub-Category'], as_index=False)[['Sales']].sum()
                          .pipe(lambda df: df[(df['year'] == 2017) & (df['Category'] == '家具')])
                          .sort_values('Sales', ascending=False), 'Sub-Category', 'Sales', '売上[USドル]', '2017家具/サブカテゴリ別売上', 0)
# 2017家具/サブカテゴリ別利益率
subcat_pr_fig = catplot_strip(merge_df.groupby(year_cat + ['Sub-Category'], as_index=False)[sls_pr].sum()
                              .assign(profit_ratio=lambda df: (df['Profit'] / df['Sales']) * 100)
                              .pipe(lambda df: df[(df['year'] == 2017) & (df['Category'] == '家具')])
                              .sort_values('profit_ratio', ascending=False), 'Sub-Category', 'profit_ratio', '利益率[%]', '2017家具/サブカテゴリ別利益率')
# 割引率と利益率の関係
disc_pr_reg_fig = reg_plot(merge_df, 'Discount', 'profit_ratio',
                           '割引率[%]', '利益率[%]', '割引率と利益率の関係')

# 2017サブカテゴリ別割引実施割合
subcat_dr_fig = bar_plot(merge_df.groupby(year_cat + ['Sub-Category'], as_index=False).agg({'Order ID': 'count', 'Discount_flg': 'sum'})
                         .assign(Discount_ratio=lambda df: (df['Discount_flg'] / df['Order ID']) * 100)
                         .pipe(lambda df: df[(df['year'] == 2017) & (df['Category'] == '家具')])
                         .sort_values('Discount_ratio', ascending=False), 'Sub-Category', 'Discount_ratio', '割引が実施される割合[%]', '2017家具/サブカテゴリ別割引実施割合', 0)
# 新規/既存顧客数の推移
cust_fig = stakedbar_plot(merge_df.groupby(['year', 'customers_new_old'], as_index=False)[
    ['Customer ID']].count(), 'year', 'Customer ID', '顧客数[人]', 'customers_new_old', '新規/既存顧客数の推移', 0)
# 新規/既存顧客の客単価
arpc_table = merge_df.groupby('customers_new_old').agg(
    {'Sales': 'sum', 'Customer ID': 'nunique'}).reset_index()
arpc_table['avg_cstm_spd'] = (
    arpc_table['Sales'] / arpc_table['Customer ID']).round(0).astype(int)
arpc_table = arpc_table[['customers_new_old', 'avg_cstm_spd']]
arpc_table.columns = ['顧客層', '客単価[ドル]']
# リピート率計算
customers_by_year = merge_df.groupby('year')['Customer ID'].unique().to_dict()
repeat_data = []
years = sorted(customers_by_year.keys())
for i in range(1, len(years)):
    prev_year = int(years[i - 1])
    curr_year = int(years[i])
    prev_customers = set(customers_by_year[prev_year])
    curr_customers = set(customers_by_year[curr_year])

    repeaters = prev_customers & curr_customers
    churners = prev_customers - curr_customers

    repeat_data.append({
        'year': curr_year,
        'type': 'リピーター',
        'count': len(repeaters)
    })
    repeat_data.append({
        'year': curr_year,
        'type': '離脱者',
        'count': len(churners)
    })
repeat_df = pd.DataFrame(repeat_data)
repeat_df['year'] = repeat_df['year'].astype(int)
print(repeat_df)
# リピート率の推移
repeat_fig = stakedbar_plot(
    repeat_df, 'year', 'count', '顧客数[人]', 'type', 'リピーターと離脱者推移', 0)
# stakedbar_plot(merge_df.groupby(['year', 'customers_new_old'], as_index=False)[
#     ['Customer ID']].count(), 'year', 'Customer ID', '顧客数[人]', 'customers_new_old', '新規/既存顧客数の推移', 0)


# %%
# plotlyグラフ描画-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 【売上・利益率】前年比比較
mry_sls_pr_fig = bar_line_plot_plotly(merge_df.groupby('year', as_index=False)[['Sales', 'Profit']].sum()
                                      .assign(profit_ratio=lambda df: (df['Profit'] / df['Sales']) * 100)
                                      .pipe(lambda df: df[df['year'].isin(df['year'].nlargest(2))]),
                                      'year', 'Sales', 'profit_ratio', '【売上・利益率】前年比比較')
# 【売上・利益率】地域別/前年比比較
mry_Reg_sls_pr_fig = stakedbar_line_plot_plotly(merge_df.groupby(['year', 'Region'], as_index=False)[['Sales', 'Profit']].sum()
                                                .assign(profit_ratio=lambda df: (df['Profit'] / df['Sales']) * 100)
                                                .pipe(lambda df: df[df['year'].isin(df['year'].drop_duplicates().nlargest(2))]), 'year', 'Sales', 'profit_ratio', 'Region', '【売上・利益率】地域別/前年比比較')
# 【売上】州別/年間
mry_st_sls_fig = bar_plot_plotly(merge_df.groupby(['year', 'State'], as_index=False)[['Sales']].sum()
                                 .pipe(lambda df: df[df['year'] == df['year'].max()])
                                 .sort_values('Sales', ascending=False), 'State', 'Sales', '売上[USドル]', '年間州別売上', 90)
# 【利益率】州別/年間
mry_st_pr_fig = catplot_strip_plotly(merge_df.groupby(['year', 'State'], as_index=False)[['Sales', 'Profit']].sum()
                                     .assign(profit_ratio=lambda df: (df['Profit'] / df['Sales']) * 100)
                                     .pipe(lambda df: df[df['year'] == df['year'].max()])
                                     .sort_values('profit_ratio', ascending=False), 'State', 'profit_ratio', '利益率[%]', '年間州別利益率')
# 【売上】月別/年間
mry_mo_sls_fig = bar_plot_plotly(merge_df.groupby(['year', 'month'], as_index=False)[['Sales', 'Profit']].sum()
                                 .assign(profit_ratio=lambda df: (df['Profit'] / df['Sales']) * 100)
                                 .pipe(lambda df: df[df['year'] == df['year'].max()]), 'month', 'Sales', '売上[USドル]', '売上・利益率の年間推移', 0)
# 【売上】日別/月間
mrm_dly_sls_fig = line_plot_plotly(merge_df.groupby(['Order Date', 'year', 'month'], as_index=False)[['Sales', 'Profit']].sum()
                                   .assign(profit_ratio=lambda df: (df['Profit'] / df['Sales']) * 100)
                                   .pipe(lambda df: df[(df['year'] == df['year'].max()) & (df['month'] == df[df['year'] == df['year'].max()]['month'].max())]), 'Order Date', 'Sales', '売上[USドル]', '売上・利益率月間推移')
# 【売上・利益率】カテゴリ別/年間
mry_cat_sls_pr_fig = stakedbar_line_plot_plotly(merge_df.groupby(['year', 'Category'], as_index=False)[['Sales', 'Profit']].sum()
                                                .assign(profit_ratio=lambda df: (df['Profit'] / df['Sales']) * 100)
                                                .pipe(lambda df: df[df['year'].isin(df['year'].drop_duplicates().nlargest(2))]), 'year', 'Sales', 'profit_ratio', 'Category', '【売上・利益率】カテゴリ別/年間')
# 【売上】サブカテゴリ別/年間
mry_subcat_sls_fig = bar_plot_plotly(merge_df.groupby(['year', 'Sub-Category'], as_index=False)[['Sales']].sum()
                                     .pipe(lambda df: df[df['year'] == df['year'].max()])
                                     .sort_values('Sales', ascending=False), 'Sub-Category', 'Sales', '売上[USドル]', '【売上】サブカテゴリ別/年間', 90)
# 【利益率】サブカテゴリ別/年間
mry_subcat_pr_fig = catplot_strip_plotly(merge_df.groupby(['year', 'Sub-Category'], as_index=False)[['Sales', 'Profit']].sum()
                                         .assign(profit_ratio=lambda df: (df['Profit'] / df['Sales']) * 100)
                                         .pipe(lambda df: df[df['year'] == df['year'].max()])
                                         .sort_values('profit_ratio', ascending=False), 'Sub-Category', 'profit_ratio', '利益率[%]', '【利益率】サブカテゴリ別/年間')
# 【割引実施割合】サブカテゴリ別/年間
mry_subcat_dr_fig = bar_plot_plotly(merge_df.groupby(['year', 'Sub-Category'], as_index=False).agg({'Order ID': 'count', 'Discount_flg': 'sum'})
                                    .assign(Discount_ratio=lambda df: (df['Discount_flg'] / df['Order ID']) * 100)
                                    .pipe(lambda df: df[df['year'] == df['year'].max()])
                                    .sort_values('Discount_ratio', ascending=False), 'Sub-Category', 'Discount_ratio', '割引が実施される割合[%]', '【割引実施割合】サブカテゴリ別/年間', 90)
# 【新規/既存顧客数】全期間
mry_cust_fig = stakedbar_plot_plotly(merge_df.groupby(['year', 'customers_new_old'], as_index=False)[
    ['Customer ID']].count(), 'year', 'Customer ID', '顧客数[人]', 'customers_new_old', '【新規/既存顧客数】年間', 0)
# 【リピート率】全期間
mry_rpt_fig = stakedbar_plot_plotly(
    repeat_df, 'year', 'count', '顧客数[人]', 'type', 'リピーターと離脱者推移', 0)
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# %%
st.set_page_config(page_title="main_dashboard", layout="wide")

# CSSによる全体書式
st.markdown("""
<style>
    .stMarkdown p { margin-bottom: 0.3rem; }
    .stMarkdown { line-height: 1.2; }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { margin-top: 0.5rem; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ページ構成選択
section = st.sidebar.radio("表示するセクションを選択", [
    "概要",
    "サマリー",
    "アジェンダ",
    "➀ 財務状況",
    "➁ 売上・利益率分析",
    "➂ 改善施策の提案",
    "➃ 今後に向けて"
])
if section == "概要":
    st.title('概要')
    st.subheader("説明対象者")
    st.text("マーケティング担当者様")
    st.subheader("取扱いデータ")
    st.text("アメリカ小売店販売のサンプルデータ")
    st.subheader("テーブル一覧")
    st.text("・発注ログ")
    st.text("・返品ログ")

elif section == "サマリー":
    graph_col, text_col = st.columns(2)
    with graph_col:
        st.write(mry_sls_pr_fig)  # 年別売上利益率
    with text_col:
        st.write(mry_mo_sls_fig)  # 月別売上/利益率
    st.write(mrm_dly_sls_fig)  # 日別売上/利益率
    st.divider()

    graph_col, text_col = st.columns([1, 2])
    with graph_col:
        st.write(mry_Reg_sls_pr_fig)  # 地域別売上/利益率
    with text_col:
        st.write(mry_st_sls_fig)  # 州別売上
        st.write(mry_st_pr_fig)  # 州別利益率
    st.divider()
    graph_col, text_col = st.columns([1, 1])
    with graph_col:
        st.write(mry_cat_sls_pr_fig)  # カテゴリ別売上/利益率
    with text_col:
        st.write(mry_subcat_sls_fig)  # サブカテゴリ別売上
        st.write(mry_subcat_pr_fig)  # サブカテゴリ別利益率
        st.write(mry_subcat_dr_fig)  # サブカテゴリ別割引実施割合
    st.divider()
    graph_col, text_col = st.columns([1, 1])
    with graph_col:
        st.write(mry_cust_fig)  # 新規/既存顧客数の推移
    with text_col:
        st.write(mry_rpt_fig)  # リピート率推移

elif section == "アジェンダ":
    # st.write("## マーケティング担当者様向け")
    # st.markdown("### <u>売上・利益構造の可視化と改善提案</u>", unsafe_allow_html=True)
    # st.markdown("#### ～サンプルスーパーストア データ分析レポート～", unsafe_allow_html=True)
    # st.write("")
    # st.divider()
    st.title('アジェンダ')
    st.subheader("➀ 財務状況")
    st.subheader("➁ 売上・利益率分析レポート")
    st.text("・月別売上")
    st.text("・カテゴリ/サブカテゴリ別 売上・利益率")
    st.text("・割引と利益率の関連性")
    st.text("・新規/既存顧客数推移と客単価")   
    st.text("・顧客のリピート率")
    st.subheader("➂ 改善施策の提案")
    st.subheader("➃ 今後に向けて")
    # st.markdown("""
    # #### ➀ 財務状況  
    # <span style='color:gray;'>・2016-17年度業績報告</span><br>
    # #### ➁ 売上・利益率分析レポート  
    # <span style='color:gray;'>・地域/州別<br>
    # ・月別売上<br>
    # ・カテゴリ/サブカテゴリ別 売上・利益率<br>
    # ・割引と利益率の関連性<br>
    # ・新規/既存顧客数推移と客単価<br>
    # ・顧客のリピート率</span><br>
    # #### ➂ 改善施策の提案  
    # #### ➃ 今後に向けて  
    # """, unsafe_allow_html=True)
    # st.divider()

# ➀ 財務状況
elif section == "➀ 財務状況":
    st.header(':blue[➀ 財務状況]')
    st.subheader("業績報告")
    graph_col, text_col = st.columns([1, 1])
    with graph_col:
        st.write(All_sls_pr_fig)
    with text_col:
        st.markdown("### :gray[売上]")
        st.markdown("### :blue[+120Kドル]")
        st.markdown("### :gray[利益率]")
        st.markdown("### :orange[-0.7%]")
    st.markdown("### <u>売上の継続的な向上と利益率の改善が必要</u>", unsafe_allow_html=True)
# ➁ 売上・利益率分析
elif section == "➁ 売上・利益率分析":
    st.header(':blue[➁ 売上・利益率分析レポート]')
    # 地域/州別 売上・利益率
    st.subheader("地域/州別 売上・利益率")
    graph_col, text_col = st.columns([1, 1])
    with graph_col:
        st.write(reg_sls_pr_fig)
    with text_col:
        st.markdown("#### :gray[西部地域が最も高い利益率を維持。]")
        st.markdown("#### :orange[南部では売上が低迷]")
        st.markdown("#### :orange[中央・南部では利益が著しく低下。]")
    graph_col, text_col = st.columns([1, 1])
    with graph_col:
        st.write(south_sls_fig)
    with text_col:
        st.markdown("##### :gray[ミシシッピ、アーカンソー、アラバマ、サウスカロライナ州では売上が低い]")
    graph_col, text_col = st.columns([1, 1])
    with graph_col:
        st.write(south_central_pr_fig)
    with text_col:
        st.markdown("##### :gray[テキサス、テネシー、ノースカロライナ、イリノイ州では赤字]")
    st.markdown("#### <u>仮説: 陳列商品が不適切、配送料が高いなど</u>", unsafe_allow_html=True)
    st.divider()
    # 月別売上
    st.subheader("月別売上")
    graph_col, text_col = st.columns([1, 1])
    with graph_col:
        st.write(mo_sls_fig)
    with text_col:
        st.markdown("#### :orange[2月が最も低い。]")
    graph_col, text_col = st.columns([1, 1])
    with graph_col:
        st.write(daily_sls_fig)
    with text_col:
        st.markdown("##### :gray[高売上日：イベントの開催]")
        st.markdown("###### :gray[(2/5スーパーボウル、2/14バレンタインデー、2/20プレジデント・デー)]")
        st.markdown("##### :gray[低売上日：急激な購買上昇による反動]")
    st.markdown("#### <u>仮説➀：天候による外出控え・物流の遅延</u>", unsafe_allow_html=True)
    st.markdown("#### <u>仮説➁：リピーターの定着率が悪い</u>", unsafe_allow_html=True)
    st.divider()

    # カテゴリ/サブカテゴリ別 売上・利益率
    st.subheader("カテゴリ/サブカテゴリ別 売上・利益率")
    graph_col, text_col = st.columns([1, 1])
    with graph_col:
        st.write(cat_sls_pr_fig)
    with text_col:
        st.markdown("#### :gray[電子機器が最も高い売上、事務用品も安定]")
        st.markdown("#### :orange[ただし、家具カテゴリが低迷]")
    graph_col, text_col = st.columns([1, 1])
    with graph_col:
        st.write(subcat_sls_fig)
    with text_col:
        st.markdown("##### :gray[本棚・インテリアの売上が低い]")
    graph_col, text_col = st.columns([1, 1])
    with graph_col:
        st.write(subcat_pr_fig)
    with text_col:
        st.markdown("##### :gray[テーブル・本棚が赤字]")
    st.markdown("#### <u>仮説：販売先の顧客層が不適切 / 割引によりコストが悪化</u>",
                unsafe_allow_html=True)
    st.divider()
    # 割引と利益率の関連性
    st.subheader("割引と利益率の関連性")
    graph_col, text_col = st.columns([1, 1])
    with graph_col:
        st.write(disc_pr_reg_fig)
    with text_col:
        st.markdown("#### :orange[割引率増加により利益率は低下]")
    graph_col, text_col = st.columns([1, 1])
    with graph_col:
        st.write(subcat_dr_fig)
    with text_col:
        st.markdown("#### :gray[椅子、テーブル、本棚は割引依存]")
        st.markdown("##### :gray[※定価で購入されずらい]")
    st.markdown("#### <u>仮説：価格と顧客層のミスマッチ</u>", unsafe_allow_html=True)
    st.divider()
    # 新規/既存顧客数推移と客単価
    st.subheader("新規/既存顧客数推移と客単価")
    graph_col, text_col = st.columns([1, 1])
    with graph_col:
        st.write(cust_fig)
    with text_col:
        st.markdown("#### :orange[全体の顧客数は増加、新規顧客は減少]")
    graph_col, text_col = st.columns([1, 1])
    with graph_col:
        st.write(arpc_table)
    with text_col:
        st.markdown("##### :gray[既存顧客は客単価が高い]")
        st.markdown("##### :orange[継続的な売上拡大には新規獲得も必要]")
    st.markdown("#### <u>仮説：集客施策が不足している</u>", unsafe_allow_html=True)
    st.divider()
    # 顧客のリピート率
    st.subheader("顧客のリピート率")
    graph_col, text_col = st.columns([1, 1])
    with graph_col:
        st.write(repeat_fig)
    with text_col:
        st.markdown("#### :orange[年々リピート率は改善傾向]")
        st.markdown("##### :gray[依然として離脱顧客は一定数存在]")
    st.markdown("#### <u>仮説：再購入促進施策に改善の余地あり</u>", unsafe_allow_html=True)
# ➂ 改善施策の提案
elif section == "➂ 改善施策の提案":
    st.header(':blue[➂ 改善施策の提案]')
    shisaku_df = pd.DataFrame(
        {
            "仮説": [
                "陳列商品が不適切、配送料が高いなど",
                "2月は天候による外出控え・物流の遅延が発生しがち",
                "急激な売上変動からリピーターの定着率が悪い",
                "販売先の顧客層が不適切",
                "割引によりコストが悪化している",
                "価格に対する顧客層のレイヤーが釣り合っていない",
                "広告/キャンペーンなどの集客が不足している",
                "再購入を促す施策に改善の余地あり"
            ],
            "改善施策": [
                "地域別に適した商品の陳列と物流コストの最適化（倉庫配置の見直しなど）",
                "2月に特化した集客施策の実施。在庫保有数の見直し",
                "再来店促進キャンペーンの展開",
                "地域/業種別ターゲティングによる販促メッセージの最適化",
                "割引率の最適化",
                "価格帯に合わせた顧客層への訴求や、高価格帯商品の価値訴求を強化",
                "新規顧客向けキャンペーンの実施とSNS等の強化",
                "ポイント制度やメールマーケティングの導入"
            ]
        }
    )
    st.dataframe(shisaku_df, use_container_width=True)
# ➃ 今後に向けて
elif section == "➃ 今後に向けて":
    st.header(':blue[➃ 今後に向けて]')
    st.markdown("#### :gray[施策実行 → 効果検証 → フィードバック → PDCAサイクルを回し最適化を図る]")
