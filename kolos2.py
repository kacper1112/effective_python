import numpy as np

# zwraca ślad macierzy 2D
def get_trace(array):
  return np.trace(array)


import numpy as np

def outer_product(vec1, vec2):
    return np.outer(vec1, vec2)

  # fcja zwraca produkt zewnętrzny dwóch wektorów

def get_common(arr1, arr2):
  # fcja zwraca unikalne wspólne elementy dwóch wektorów (te same elementy na tych samych pozycjach)
  return np.unique(arr1[arr1 == arr2])

  import pandas as pd

def series_uppercase(series):
    return series.str.capitalize()
  # zwraca Series, w którym każdy element zaczyna się wielką literą


class FunCount(object):
    # uzupełnij kod klasy tak, żeby obiekt "dekorował" funkcje i liczył jej wywołania
    cnt = 0
    def __init__(self, f):
        self.cnt = 0
        self.f = f
    def __call__(self, *args):
        self.cnt = self.cnt + 1
        return sum([arg for arg in args])
    def __str__(self):
        return '"fun" was called {} times.'.format(self.cnt)

def add_min_max(df):
   df['max'] = df[[0, 1, 2, 3]].max(axis=1)
   df['min'] = df[[0, 1, 2, 3]].min(axis=1)
  # fcja dodaje kolumny 'min' oraz 'max' zawierające minumum i maksimum każdego wiersza
ZAPIERDALA


# zwraca elementy serii na pozycjach pos

def get_series_at_positions(series, pos):
  return series.loc[pos]

  class A(object):
    # uzupełnij kod klasy tak, aby w polu cnt była liczba utworzonych obiektów tej klasy
    cnt = 0
    def __init__(self):
        A.cnt += 1
        self.id = A.cnt



# narysuj wykres oraz zaznacz minima i maksima lokalne tak jak na rysunku
from scipy.signal import argrelextrema
def plot_points(X):
  df = pd.DataFrame(X, columns=['data'])
  n = 2
  df['min'] = df.iloc[argrelextrema(df.data.values, np.less_equal,
                    order=n)[0]]['data']
  df['max'] = df.iloc[argrelextrema(df.data.values, np.greater_equal,
                    order=n)[0]]['data']

  # Plot results

  plt.scatter(df.index, df['min'], c='r')
  plt.scatter(df.index, df['max'], c='g')
  plt.plot(df.index, df['data'])
  plt.show()

steps = [-1,+1]
walk = np.random.choice(steps, 200).cumsum()

plot_points(walk)

# niby lepsze od tego wyzej
def rescale(X):
    return X / X.sum(axis=1)[:, np.newaxis]



# def plot_log(x, y):

#     plt.plot(x,y)
#     plt.yscale('log')
#     plt.show()

def plot_log(x, y):
  plt.plot(x,y)
  plt.xlabel('x')
  plt.ylabel('$e^x$')
  plt.yscale('log')
  plt.show()




def get_union(arr1, arr2):
  # fcja zwraca iloczyn mnogościowy dwóch wektorów
  return list(set(arr1).intersection(arr2))


def mirror(arr):
    return np.flip(arr, axis=None)
  # fcja odbija elementy macierzy wzgl. "drugiej" przekątnej.




mport pandas as pd

def series_difference(series1, series2):
  # zwraca różnicę mnogościową dwóch serii
  result = []
  for i in series1.values:
    if i not in series2.values:
      result.append(i)
      
  return result


def mirror(arr):
  # fcja odbija elementy macierzy wzgl. "drugiej" przekątnej.
  return np.rot90(np.rot90(np.rot90(np.fliplr(arr))))





