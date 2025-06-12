import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
     14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
Y = [31.58, 38.11, 40.42, 38.93, 34.60, 32.38, 41.34, 29.45, 38.70, 39.68, 34.70, 38.24, 28.31,
     35.27, 37.30, 35.39, 37.84, 32.80, 34.11, 37.73, 33.75, 35.69, 42.30, 31.72]
fig = plt.figure()
plt.bar(X, Y, 0.4, color="green")
plt.xlabel("Kodak No.")
plt.ylabel("PSNR")
plt.title("Test results on Kodak dataset")


plt.show()
plt.savefig("barChart.jpg")
