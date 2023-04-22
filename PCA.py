import pandas as pd
import sklearn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# CSVファイルを読み込む
weather_data = pd.read_csv("weather_data.csv", index_col=0)

print(weather_data.head(5))

setumei = weather_data.iloc[:,[0,2,3,4]]
target = weather_data.iloc[:,1]

pca = PCA( whiten=True)
pca.fit(setumei)

loadings = pd.DataFrame(pca.components_.T, index=setumei.columns)
print(loadings)

score = pd.DataFrame(pca.transform(setumei), index=setumei.index)
print(score)

contribution = pd.DataFrame(pca.explained_variance_ratio_)
print(contribution)

# 累積寄与率を算出（.cusum()で累積和を計算 .sum()では総和しか得られない）
sum_contribution = contribution.cumsum()
print(sum_contribution)

cont_cumcont_ratios = pd.concat([contribution, sum_contribution], axis=1).T

cont_cumcont_ratios.index = ['contribution_ratio', 'cumulative_contribution_ratio']  # 行の名前を変更

# グラフを横に並べて描画
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# 寄与率を棒グラフで、累積寄与率を線で入れたプロット図を重ねて描画
x_axis = range(1, contribution.shape[0] + 1)  # 1 から成分数までの整数が x 軸の値
axs[0].bar(x_axis, contribution.iloc[:, 0], align='center')  # 寄与率の棒グラフ
axs[0].plot(x_axis, sum_contribution.iloc[:, 0], 'r.-')  # 累積寄与率の線を入れたプロット図
axs[0].set_xlabel('Number of principal components')  # 横軸の名前
axs[0].set_ylabel('Contribution ratio(blue),\nCumulative contribution ratio(red)')  # 縦軸の名前
axs[0].tick_params(labelsize=14)  # 軸のラベルのフォントサイズを設定

# 第 1 主成分と第 2 主成分の散布図 (band_gap の値でサンプルに色付け)
sc = axs[1].scatter(score.iloc[:, 0], score.iloc[:, 1], c=weather_data.iloc[:, 1], cmap=plt.get_cmap('jet'))
axs[1].set_xlabel('t1')
axs[1].set_ylabel('t2')
axs[1].set_title('t1-t2')

clb = plt.colorbar(sc, ax=axs[1])
clb.set_label('precipitation', labelpad=-20, y=1.1, rotation=0)
axs[1].tick_params(labelsize=14)  # 軸のラベルのフォントサイズを設定

plt.tight_layout()  # グラフのレイアウトを調整
plt.show()
#plt.savefig("result3.png")