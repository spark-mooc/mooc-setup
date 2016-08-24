w = [0.14861399988664878,0.2632357078685982,0.009135268543963385,0.30988066494981475,0.1251216042128011,0.25666203741508375,]
theta = 1.0
var('x,y,z')
p1 = implicit_plot3d((w[0] * x**2 + w[1] * y**2 + w[2] * z**2 + w[3] * sqrt(2) * x*y) + (w[4] * sqrt(2) * x*z + w[5] * sqrt(2) * y*z) == theta, (x, 0, 1), (y, 0, 1), (z, 0, 1))
p2 = point3d(negs,size=10,color='red')
p3 = point3d(poss,size=10,color='blue')
show(p1+p2+p3)