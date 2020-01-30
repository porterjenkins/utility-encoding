import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


movie_logit = pd.read_csv("tuning-movie-logit.csv")

fig,ax=plt.subplots()
ax.set_xlabel("lambda",fontsize=14)
ax.set_ylabel("HR@5",color="blue",fontsize=14)
ax.set_ylim([.5,1])
ax.plot(movie_logit['lambda'], movie_logit['Hit Ratio'], marker='o', color='b')

ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(movie_logit['lambda'],  movie_logit['NDCG'],color="red",marker="o")
ax2.set_ylabel("NCDG@5",color="red",fontsize=14)
ax2.set_ylim([.5,1])

plt.savefig("tuning-movie-logit.pdf")