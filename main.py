import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from fast_hough_transform import fast_hough_transform


def transform_all_images():
    '''
    Поворачивает все изображения на нужный угол, используя два варианта интерполяции
    :return: None
    '''
    for i in range(1, 11):
        gray_image = cv2.imread(os.path.join('images', f'{i}.jpg'), cv2.IMREAD_GRAYSCALE)
        gray_image = cv2.Canny(gray_image, 150, 250)
        height, width = gray_image.shape
        transform, _, _ = fast_hough_transform(gray_image)
        max_var_row = np.argmax(np.var(transform, axis=0))
        angle = np.degrees(np.arctan((max_var_row - transform.shape[1] // 2) / gray_image.shape[1]))

        original_image = cv2.imread(os.path.join('images', f'{i}.jpg'), cv2.IMREAD_COLOR)
        center = (int(width / 2), int(height / 2))
        rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1)
        transformed_image = cv2.warpAffine(original_image, rotation_matrix, (width, height),
                                           borderMode=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join('transformed_images', f'{i}_bilinear.jpg'), transformed_image)
        transformed_image_2 = cv2.warpAffine(original_image, rotation_matrix, (width, height),
                                             borderMode=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join('transformed_images', f'{i}_nearest.jpg'), transformed_image_2)


def measure_time_and_memory():
    '''
    Замеряет зависимость времени работы алгоритма от площади изображения,
    а также времени и требуемой памяти от длины/ширины
    :return: None
    '''
    time_algo = np.zeros(120)
    mem_algo = np.zeros(120)
    gray_image = cv2.imread(os.path.join('images', f'5.jpg'), cv2.IMREAD_GRAYSCALE)
    gray_image = cv2.Canny(gray_image, 150, 250)
    sizes = np.arange(10, 1201, 10)
    for size in sizes:
        _, time_algo[size // 10 - 1], mem_algo[size // 10 - 1] = fast_hough_transform(gray_image[:size, :size])

    plt.figure(figsize=(12, 7))
    plt.plot(sizes ** 2, time_algo)
    plt.title('Зависимость времени работы алгоритмов от площадь изображения')
    plt.xlabel('Площадь изображения')
    plt.ylabel('Время работы (мсек/мегапиксель)')
    plt.grid()
    plt.savefig('time_from_area.png')
    plt.show()

    plt.figure(figsize=(12, 7))
    plt.plot(sizes, time_algo * sizes ** 2 / 1e9)
    plt.title('Зависимость времени работы алгоритмов от длины/ширины')
    plt.xlabel('N')
    plt.ylabel('Время работы')
    plt.grid()
    plt.savefig('time_from_N.png')
    plt.show()

    plt.figure(figsize=(12, 7))
    plt.plot(sizes, mem_algo)
    plt.title('Зависимость потребляемой памяти от длины/ширины')
    plt.xlabel('N')
    plt.ylabel('Потребляемая память (в байтах)')
    plt.grid()
    plt.savefig('memory.png')
    plt.show()


if __name__ == '__main__':
    transform_all_images()
    # thread = threading.Thread(target=measure_time_and_memory)
    # thread.start()
    # thread.join()
