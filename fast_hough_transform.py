import time

import numpy as np
from numba import jit


@jit(nopython=True)
def recursive_hough(image, left, right, memory_obj):
    '''
    :param image: исходное серое изображение с 1 каналом
    :param left: левая граница части изображения
    :param right: правая граница части изображения
    :return: преобразование Хафа для заданной части изображения
    '''
    image_height = image.shape[0]
    width = right - left + 1
    transform = np.zeros((image_height, width))
    memory_obj[0] += transform.size * 8
    memory_obj[1] = max(memory_obj[0], memory_obj[1])
    if right - left == 1:
        transform[:, 0] = image[:, left]
    else:
        mid = right - (right - left) // 2
        left_transform = recursive_hough(image, left, mid, memory_obj)
        right_transform = recursive_hough(image, mid, right, memory_obj)
        for y in range(image_height):
            for shift in range(width):
                transform[y, shift] = left_transform[y, shift // 2] + \
                                      right_transform[
                                          (y + shift // 2 + shift % 2) % image_height, shift // 2 - shift % 2]
        memory_obj[0] -= left_transform.size * 8
        memory_obj[0] -= right_transform.size * 8
    return transform


def fast_hough_transform(image):
    '''
    :param image: исходное серое изображение с 1 каналом
    :return: преобразование Хафа изображения
    '''
    start_time = time.time()
    image_width = image.shape[1]

    # для корректной обработки в numba используем массив numpy, первый элемент которого
    # содержит текущий размер занимаемой памяти, а второй - максимальный
    memory_obj = np.zeros(2)

    # рассматриваем как положительные, так и отрицательные углы
    transform = recursive_hough(image, 0, image_width, memory_obj)
    flip_transform = recursive_hough(np.flip(image, axis=1), 0, image_width, memory_obj)

    algorithm_time = time.time() - start_time

    return np.hstack(
        [np.flip(transform, axis=1), flip_transform[:, 1:]]), algorithm_time * 1e9 / image.size, memory_obj[1]
