from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import pandas as pd
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)

# PANDAS ========================================================================
r_cols = ["user id" , "movie id" , "rating", "timestamp"]
ratings = pd.read_csv('./u.data', sep='\t', names=r_cols, encoding='latin-1')

ratings = ratings.sort_values("user id")
ratings = ratings.loc[(ratings["rating"] >= 4)]
# print(ratings)
# print(len(ratings))

# Extract data using pandas
m_cols = ["movie id" , "movie title", "release date" , "video release date" ,
              "IMDb URL" , "unknown" , "Action" , "Adventure" , "Animation" ,
              "Children's" , "Comedy" , "Crime" , "Documentary"
              , "Drama" , "Fantasy" ,
              "Film-Noir" , "Horror" , "Musical" , "Mystery" , "Romance" , "Sci-Fi" ,
              "Thriller" , "War" , "Western"]

movies = pd.read_csv('./u.item', sep='|', names=m_cols, encoding='latin-1')
movies = movies.iloc[:, [0] + list(range(6, 24))]
# print(movies)

newArr = pd.merge(ratings,movies, on='movie id')
newArr = newArr.sort_values("user id")
newArrOnlyRatings = newArr.iloc[:, [0] + list(range(4, 22))]
# print(newArrOnlyRatings)

# Data sum values
result = newArrOnlyRatings.groupby(["user id"]).sum()
result = result.iloc[:, list(range(0, 18))]
data = result.values.tolist()
# print(data[0])
# end Data sum values

# END PANDAS ========================================================================


hierachical = linkage(data, method = 'median', metric = 'euclidean')

plt.figure(figsize = (19, 9))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    hierachical,
    leaf_rotation = 90,
    leaf_font_size = 10,
    truncate_mode = 'lastp',
    p = 100
)

plt.show()
