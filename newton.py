import argparse 

parser = argparse.ArgumentParser(description='Miejse zerowe costam.')

parser.add_argument('func', type=str, help='function')
parser.add_argument('-x', type=int, help='punkt startowy')
parser.add_argument('-k', type=int, help='wielkosc kroku w pochodnej')
parser.add_argument('-n', type=int, help='ilosc krokow')
parser.add_argument('-e', type=float, help='dokladnosc')
parser.add_argument('-hh', help='help')

args = parser.parse_args()

x,k,n,e = args.x, args.k, args.n, args.e

dupa = """
def func(*args):
    x=10
    return x**2+10

"""

print(eval(dupa))
