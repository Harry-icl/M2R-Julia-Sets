from julia import CubicMap
import pickle

threebrot = CubicMap(a=0)
mndl = threebrot.draw_mandelbrot()
threebrot.b = 0
jul = threebrot.draw_julia()

file = open('./tests/data/threebrot_mandelbrot.pkl', 'wb')
pickle.dump(mndl, file)
file.close()

file = open('./tests/data/threebrot_julia.pkl', 'wb')
pickle.dump(jul, file)
file.close()