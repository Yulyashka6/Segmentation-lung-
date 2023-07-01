import numpy as np
import pandas as pd
import pydicom
import os
import scipy.ndimage as ndimage
from skimage import measure, morphology, segmentation
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import PIL

start = datetime.now()

INPUT_FOLDER = "/Diplom/train/"

patients = os.listdir(INPUT_FOLDER)
patients.sort()

print("Some examples of patient IDs:")
print(",\n".join(patients[:5]))

def load_scan(path):
# Загружает сканы из папки в список.
# Параметры: path (путь к папке)
# Возвращает: срезы (список срезов)

    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))  

    return slices

def get_pixels_hu(scans):
    #Преобразует необработанные изображения в единицы измерения Хаунсфилда (HU).
    #Параметры: сканирование (необработанные изображения)
    #Возвращает: изображение (массив NumPy)
    
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)

    # Значения изображения вне зоны сканирования зануляем
    image[image == -2000] = 0
    
    # HU = m*P + b
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

def generate_markers(image):
    # Генерируем маркеры для заданного изображения
    # Параметр: заданное изображение
    # Возвращает: Внутренние, внешние маркеры и маркер для водораздела
    # Фомирование внутренних макеров
    marker_internal = image < -680
    marker_internal = segmentation.clear_border(marker_internal)
    marker_internal_labels = measure.label(marker_internal)
    
    areas = [r.area for r in measure.regionprops(marker_internal_labels)]
    areas.sort()
    
    if len(areas) > 2:
        for region in measure.regionprops(marker_internal_labels):
            if region.area < areas[-2]:
                for coordinates in region.coords:                
                    marker_internal_labels[coordinates[0], coordinates[1]] = 0
    
    marker_internal = marker_internal_labels > 0
    
    # Создание внешних маркеров
    external_a = ndimage.binary_dilation(marker_internal, iterations=10)
    external_b = ndimage.binary_dilation(marker_internal, iterations=55)
    marker_external = external_b ^ external_a
    
    # Создание маркеров для водораздела
    marker_watershed = np.zeros((512, 512), dtype=np.int64)
    marker_watershed += marker_internal * 255
    marker_watershed += marker_external * 128
    
    return marker_internal, marker_external, marker_watershed
    # return marker_internal, external_a, marker_watershed

def seperate_lungs(image):
# Сегментируем легкие с использованием метода водораздела
# Параметры: изображение (сканируемое изображение) 
# Возвращает: 
#    - Легкие, сегментированные с помощью водораздела
#    - Градиент Собеля
    
    marker_internal, marker_external, marker_watershed = generate_markers(image)
    
    # Создание градиента Собеля
    
    sobel_filtered_dx = ndimage.sobel(image, 1)
    sobel_filtered_dy = ndimage.sobel(image, 0)
    sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)
    sobel_gradient *= 255.0 / np.max(sobel_gradient)
    
    # Алгоритм водораздела 
    # Передаем изображение, свернутое 
    # при помощи оператора Собела, и созданные маркеры для водораздела. 
    # Получаем: изображение, размеченное с помощью алгоритма водораздела
    
    watershed = segmentation.watershed(sobel_gradient, marker_watershed)
    marker_internal = np.uint8(marker_internal)
    edge, _ = cv2.findContours(marker_internal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edge_based = cv2.drawContours(marker_internal, edge, -1, (255,255,255), -1)
    
    
    return sobel_gradient, watershed, edge_based


def apply_a_contour(image, labels, edge):
    contours_watershed = measure.find_contours(labels)
    contours_edge = measure.find_contours(edge)
    for contour in contours_watershed:
        plt.plot(contour[:,1], contour[:,0], linewidth=1, color = '#e3fc03')
    
    plt.imshow(image, cmap = 'gray')
    plt.title('Result 1')
    plt.show()

    for contour in contours_edge:
        plt.plot(contour[:,1], contour[:,0], linewidth=1, color = '#3dfc03')

    plt.imshow(image, cmap = 'gray')
    plt.title('Result 2')
    plt.show()
    # labels1 = np.uint8(labels)
    # contours, ierarhy = cv2.findContours(labels1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # imagewithcontour = cv2.drawContours(image1, contours, -1, (0,255,0), 2)
    
    # plt.imshow(imagewithcontour, cmap = 'gray')
    # plt.title('Result')
    # plt.show()

#  Подгружаем пример оригинального изображения
# Пациент: 24, срез: 78
test_patient_scans = load_scan(INPUT_FOLDER + patients[30])
test_patient_images = get_pixels_hu(test_patient_scans)
number_patient_skan = test_patient_images[134]
plt.imshow(number_patient_skan, cmap='gray')
plt.title("Original Slice")
# plt.show()

# Проверка работоспособности функции по созданию маркеров
test_patient_internal, test_patient_external, test_patient_watershed = generate_markers(number_patient_skan)

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(5,5))

ax1.imshow(test_patient_internal, cmap='gray')
ax1.set_title("Internal Marker")
ax1.axis('off')

ax2.imshow(test_patient_external, cmap='gray')
ax2.set_title("External Marker")
ax2.axis('off')

ax3.imshow(test_patient_watershed, cmap='gray')
ax3.set_title("Watershed Marker")
ax3.axis('off')

plt.show()

# Проверка работы алгоритма водораздела
test_sobel_gradient, test_watershed, test_edge_based = seperate_lungs(number_patient_skan)
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(5,5))
# saving_watershed = cv2.imwrite(r'C:\Diplom\Result\Patient_2\3.jpg', test_watershed)
saving_edge = cv2.imwrite(r"C:\Diplom\Result_edge_segmentation\Patient_1\14.jpg", test_edge_based)
ax1.imshow(test_sobel_gradient, cmap='gray')
ax1.set_title("Sobel")
ax1.axis('off')

ax2.imshow(test_watershed, cmap='gray')
ax2.set_title("Watershed")
ax2.axis('off')

ax3.imshow(test_edge_based, cmap='gray')
ax3.set_title("Edge")
ax3.axis('off')

plt.show()
# print(test_watershed.dtype)
# print(test_watershed.shape)

result = apply_a_contour(number_patient_skan, test_watershed, test_edge_based)
end = (datetime.now() - start)
print(end)









