from PIL import Image
import numpy as np
from MedianFilters import SimpleMedianFilter, HuangMedianFilter, ConstantTimeMedianFilter
from funcs import check_equality, speed_bechmark
import matplotlib.pyplot as plt

image = np.array(Image.open("data/lena.jpg"))

simple_median_filter = SimpleMedianFilter(image)
huang_median_filter = HuangMedianFilter(image)
constant_time_median_filter = ConstantTimeMedianFilter(image)

simple_median_filter.process(3)
huang_median_filter.process(3)
constant_time_median_filter.process(3)

Image.fromarray(simple_median_filter.processed).save('lena_processed.jpg')
Image.fromarray(huang_median_filter.processed).save('lena_processed_huang.jpg')
Image.fromarray(constant_time_median_filter.processed).save('lena_processed_constant.jpg')

res1 = np.array(Image.open('lena_processed.jpg'))
res2 = np.array(Image.open('lena_processed_huang.jpg'))
res3 = np.array(Image.open('lena_processed_constant.jpg'))

check_equality([res1, res2, res3])

R_grid = np.arange(3, 100, 5)

res_1 = speed_bechmark(image, SimpleMedianFilter)
res_2 = speed_bechmark(image, HuangMedianFilter)
res_3 = speed_bechmark(image, ConstantTimeMedianFilter)

plt.plot(R_grid, res_1, color='red', label='Sorting')
plt.plot(R_grid, res_2, color='green', label='Huang')
plt.plot(R_grid, res_3, color='blue', label='Constant time')
plt.xlabel("R")
plt.ylabel("мсек/мегаПиксель")
plt.legend()
plt.savefig('benchmark.png')