import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def binary_image(image):
  mask = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)[1]
  return mask

list_dice_score_watershed = []
for i in range(0, 21):
  # Загрузка изображений
  i = str(i)
  reference_lung = cv2.imread("c:/Diplom/My_dataset/Patient_2/"+ i +".jpg", cv2.IMREAD_GRAYSCALE)
  segmented_lung = cv2.imread("c:/Diplom/Result/Patient_2/"+ i +".jpg", cv2.IMREAD_GRAYSCALE)

  # Преобразуем в бинарные маски
  binary_reference_lung =binary_image(reference_lung)
  binary_segmented_lung =binary_image(segmented_lung)
  # # Check
  # cv2.imshow('binary_reference', binary_reference_lung)
  # cv2.imshow('binary_segmented', binary_segmented_lung)
  # cv2.waitKey()

  # Расчет коэффициентов Дайса
  difference_mask = cv2.bitwise_and(binary_reference_lung, binary_segmented_lung)
  count_difference_mask = cv2.countNonZero(difference_mask)
  not_zero_reference = cv2.countNonZero(binary_reference_lung)
  not_zero_segmented = cv2.countNonZero(binary_segmented_lung)
  dice_score = 2 * count_difference_mask / (not_zero_reference + not_zero_segmented)
  list_dice_score_watershed.append(dice_score)
  
list_dice_score_edge = []
for i in range(0, 21):
  # Загрузка изображений
  i = str(i)
  reference_lung = cv2.imread("c:/Diplom/My_dataset/Patient_2/"+ i +".jpg", cv2.IMREAD_GRAYSCALE)
  segmented_lung = cv2.imread("c:/Diplom/Result_edge_segmentation/Patient_2/"+ i +".jpg", cv2.IMREAD_GRAYSCALE)

  # Преобразуем в бинарные маски
  binary_reference_lung =binary_image(reference_lung)
  binary_segmented_lung =binary_image(segmented_lung)
  # # Check
  # cv2.imshow('binary_reference', binary_reference_lung)
  # cv2.imshow('binary_segmented', binary_segmented_lung)
  # cv2.waitKey()

  # Расчет коэффициентов Дайса
  difference_mask = cv2.bitwise_and(binary_reference_lung, binary_segmented_lung)
  count_difference_mask = cv2.countNonZero(difference_mask)
  not_zero_reference = cv2.countNonZero(binary_reference_lung)
  not_zero_segmented = cv2.countNonZero(binary_segmented_lung)
  dice_score = 2 * count_difference_mask / (not_zero_reference + not_zero_segmented)
  list_dice_score_edge.append(dice_score)
  
tochn = sum(list_dice_score_watershed)/(len(list_dice_score_watershed))
print(tochn)
tochn_1 = sum(list_dice_score_edge)/(len(list_dice_score_edge))
print(tochn_1)
# x = [f'{i}' for i in range(21)]
# y = list_dice_score_edge
# plt.bar(x, y , color = '#485696', label = 'Водораздел')
# # plt.bar(x, y1, color = '#989FBF', label = 'Контурный метод')
# plt.axis([0, 20, min(y)-0.015, max(y)])
# plt.title('Оценка эффективности алгоритма с помощью метрики Дайса')
# # deep_purple = mpatches.Patch(color = '#485696', label = 'Водораздел')
# # lite_purple = mpatches.Patch(color = '#989FBF', label = 'Контурный метод')
# plt.legend()
# plt.show()
x = [f'{i}' for i in range(21)]
y = list_dice_score_watershed
y1 = list_dice_score_edge
plt.bar(x, y , color = '#485696', label = 'Водораздел')
plt.bar(x, y1, color = '#989FBF', label = 'Контурный метод')
plt.axis([0, 20, min(y1), max(y)])
plt.title('Оценка эффективности алгоритма с помощью метрики Дайса')
# deep_purple = mpatches.Patch(color = '#485696', label = 'Водораздел')
# lite_purple = mpatches.Patch(color = '#989FBF', label = 'Контурный метод')
plt.legend()
plt.show()