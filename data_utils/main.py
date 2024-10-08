import math
import os
import random

import cv2 as cv
import numpy as np


def pad_img(img, height: int, width: int):
    old_h, old_w = img.shape[0], img.shape[1]
    # If height is less than height pad to height.
    if old_h < height:
        to_pad = np.zeros((height - old_h, old_w)) * height
        img = np.concatenate((img, to_pad))
        new_height = height
    else:
        new_height = old_h
    # If width is less than width pad to width.
    if old_w < width:
        to_pad = np.zeros((new_height, width - old_w)) * 255
        img = np.concatenate((img, to_pad), axis=1)
        new_width = width
    return img


def roundup(x):
    return int(math.ceil(x / 10.0)) * 10


line_img_paths = sorted(os.listdir(f"data_processed/lines/images"))
line_txt_paths = sorted(os.listdir(f"data_processed/lines/text"))

word_img_paths = sorted(os.listdir(f"data_processed/words/images"))
word_txt_paths = sorted(os.listdir(f"data_processed/words/text"))
# Outliers
words_skip = [
    "10003008522",
    "10003019841",
    "10001010904",
    "10003008522",
    "10003002911",
    "10003019841",
    "10003017056",
    "10001003114",
    "10003002913",
    "10003002912",
    "10003008521",
]

line_ids = []
for line_img_path in line_img_paths:
    split_path = os.path.split(line_img_path)
    file_name = split_path[1]
    file_idx = os.path.splitext(file_name)[0]
    line_ids.append(file_idx)

word_ids = []
for word_img_path in word_img_paths:
    split_path = os.path.split(word_img_path)
    file_name = split_path[1]
    file_idx = os.path.splitext(file_name)[0]
    word_ids.append(file_idx)

characters = {}

# Line processing.
max_line_img_width = 0
max_line_img_height = 0
max_line_char_length = 0
sum_line_char_size_ratio = 0.0

file_lines_test = open("data_processed/lines/annotation_test.txt", "a+")
file_lines_val = open("data_processed/lines/annotation_val.txt", "a+")

for i, line_id in enumerate(line_ids):
    line_img = cv.imread(f"data_processed/lines/images/{line_id}.png")
    line_img = cv.cvtColor(line_img, cv.COLOR_BGR2GRAY)
    ret, line_img = cv.threshold(line_img, 100, 255, cv.THRESH_BINARY)

    rand_num: int = random.randint(1, 10)

    line_img = pad_img(line_img, 69, 1351)
    # cv.imshow("padded_line", line_img)
    # cv.waitKey(0)

    line_txt = ""
    with open(f"data_processed/lines/text/{line_id}.txt", "r") as f:
        line_txt = f.readline()

    h, w = line_img.shape
    max_line_img_width = max(max_line_img_width, w)
    max_line_img_height = max(max_line_img_height, h)

    num_chars = len(line_txt)
    max_line_char_length: int = max(max_line_char_length, num_chars)
    line_char_size_ratio = num_chars / ((w * h) * 1.0)
    sum_line_char_size_ratio += line_char_size_ratio

    # Outlier detection
    # line_avg = 0.0006308517615384173
    # if line_char_size_ratio > (line_avg * 2) or line_char_size_ratio < (line_avg / 2):
    #     print(line_id)
    #     print(line_txt)
    #     cv.imshow("line", line_img)
    #     cv.waitKey(0)

    for char in line_txt:
        if char in characters:
            characters[char] = characters[char] + 1
        else:
            characters[char] = 1

    if rand_num > 8:
        # write to annotations.txt!
        file_lines_val.writelines(f"{line_id}.png  _ _  {line_txt}\n")
    else:
        file_lines_test.writelines(f"{line_id}.png  _ _  {line_txt}\n")

file_lines_test.close()
file_lines_val.close()
avg_line_char_size_ratio = sum_line_char_size_ratio / len(line_ids)
print("Max Line Image Width:", max_line_img_width)  # 1351px
print("Max Line Image Height:", max_line_img_height)  # 69px
print("Max Line Character Length:", max_line_char_length)
print("Average line char:size ratio:", avg_line_char_size_ratio)

print(sorted(characters))

# Word processing.
max_word_img_width = 0
max_word_img_height = 0
max_word_char_length = 0
sum_word_char_size_ratio = 0.0

file_words_test = open("data_processed/words/annotation_test.txt", "a+")
file_words_val = open("data_processed/words/annotation_val.txt", "a+")

for i, word_id in enumerate(word_ids):
    if word_id in words_skip:
        continue

    rand_num: int = random.randint(1, 10)

    word_img = cv.imread(f"data_processed/words/images/{word_id}.png")
    word_img = cv.cvtColor(word_img, cv.COLOR_BGR2GRAY)
    ret, word_img = cv.threshold(word_img, 100, 255, cv.THRESH_BINARY)

    word_img = pad_img(word_img, 69, 598)
    # cv.imshow("padded_word", word_img)
    # cv.waitKey(0)

    word_txt = ""
    with open(f"data_processed/words/text/{word_id}.txt", "r") as f:
        word_txt = f.readline()

    h, w = word_img.shape
    max_word_img_width = max(max_word_img_width, w)
    max_word_img_height = max(max_word_img_height, h)

    num_chars = len(word_txt)
    max_word_char_length: int = max(max_word_char_length, num_chars)
    word_char_size_ratio = num_chars / ((w * h) * 1.0)

    if rand_num > 8:
        # write to annotations.txt!
        file_words_val.writelines(f"{word_id}.png  _ _  {word_txt}\n")
    else:
        file_words_test.writelines(f"{word_id}.png  _ _  {word_txt}\n")

    # Outlier detection
    # word_avg = 0.0006329150858485357
    # if word_char_size_ratio > (word_avg * 2) or word_char_size_ratio < (word_avg / 2):
    #     print(word_id, word_txt)
    #     print(word_txt)
    #     cv.imshow("word", word_img)
    #     cv.waitKey(0)

    sum_word_char_size_ratio += word_char_size_ratio

file_words_test.close()
file_words_val.close()
avg_word_char_size_ratio = sum_word_char_size_ratio / len(word_ids)

print("Max Word Image Width:", max_word_img_width)  # 598px
print("Max Word Image Height:", max_word_img_height)  # 69px
print("Max Word Character Length:", max_word_char_length)
print("Average word char:size ratio:", avg_word_char_size_ratio)
