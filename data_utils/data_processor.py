import os
from typing import List, Tuple

import cv2 as cv
import pysrt
from pysrt.srtitem import SubRipItem

dir_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.split(dir_path)[0]
data_raw_path = os.path.join(root_path, "data_raw")
data_processed_path = os.path.join(root_path, "data_processed")

movies = dict(
    ready_player_one="10000",
    scream="10001",
    john_wick="10002",
    the_social_network="10003",
    ace_ventura_pet_detective="10004",
)


for movie_title, movie_id in movies.items():
    print(movie_id, movie_title)

    # Load the movie's subtitle images.
    imgs_path = os.path.join(data_raw_path, "images", movie_title)
    imgs_paths = sorted(os.listdir(imgs_path))
    imgs = [cv.imread(os.path.join(imgs_path, img_path)) for img_path in imgs_paths]

    # Load the movie's srt file (text-based subtitles).
    srts = pysrt.open(os.path.join(data_raw_path, "srts", movie_title + ".srt"))

    # Ensure same number of images as subtitles in srt.
    assert len(imgs) == len(srts)

    for i in range(len(srts)):
        srt: SubRipItem = srts[i]
        srt_lines: str = srt.text_without_tags.splitlines()

        img = imgs[i]

        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, thresh2 = cv.threshold(img_gray, 100, 255, cv.THRESH_BINARY)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (150, 8))
        mask = cv.morphologyEx(thresh2, cv.MORPH_DILATE, kernel)

        bboxes = []
        bboxes_img = img.copy()
        contours = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_L1)
        contours = contours[0] if len(contours) == 2 else contours[1]
        for cntr in contours:
            x, y, w, h = cv.boundingRect(cntr)
            cv.rectangle(bboxes_img, (x, y), (x + w, y + h), (0, 0, 255), 1)
            bboxes.append((x, y, w, h))

        # Sort bounding boxes by y-coordinate to order them top-to-bottom.
        bboxes: List[Tuple[int, int, int, int]] = sorted(
            bboxes, key=lambda bbox: bbox[1]
        )

        for j, bbox in enumerate(bboxes):
            (x, y, w, h) = bbox
            bbox_img = img[y : y + h, x : x + w]
            bbox_img_inv = cv.bitwise_not(bbox_img.copy())
            ret, bbox_img_inv_bin = cv.threshold(
                bbox_img, 100, 255, cv.THRESH_BINARY_INV
            )

            sub_id = movie_id + str(i + 1) + str(j).zfill(5)

            imgp = os.path.join(data_processed_path, "lines", "images", sub_id + ".png")
            txtp = os.path.join(data_processed_path, "lines", "text", sub_id + ".txt")
            cv.imwrite(imgp, bbox_img_inv)
            with open(txtp, "w") as f:
                f.write(srt_lines[j])

            line_txt = srt_lines[j]

            line_img = bbox_img.copy()
            line_img_gray = cv.cvtColor(line_img, cv.COLOR_BGR2GRAY)
            ret, word_thresh = cv.threshold(line_img_gray, 100, 255, cv.THRESH_BINARY)

            word_kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 12))
            word_mask = cv.morphologyEx(word_thresh, cv.MORPH_DILATE, word_kernel)

            # Split line into words
            word_bboxes = []
            word_bboxes_img = line_img.copy()
            word_contours = cv.findContours(
                word_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_L1
            )

            word_contours = (
                word_contours[0] if len(word_contours) == 2 else word_contours[1]
            )

            for wcntr in word_contours:
                x, y, w, h = cv.boundingRect(wcntr)
                cv.rectangle(word_bboxes_img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                word_bboxes.append((x, y, w, h))

            words = line_txt.split(" ")

            if len(words) == len(word_bboxes):
                # Sort the word contours by x coordinate to order them LTR.
                word_bboxes: List[Tuple[int, int, int, int]] = sorted(
                    word_bboxes, key=lambda bbox: bbox[0]
                )
                for l, wbbox in enumerate(word_bboxes):
                    (x, y, w, h) = wbbox
                    wbbox_img = line_img[y : y + h, x : x + w]
                    ret, wbbox_img_inv_bin = cv.threshold(
                        wbbox_img, 100, 255, cv.THRESH_BINARY_INV
                    )

                    word = words[l]
                    wimgp = os.path.join(
                        data_processed_path, "words", "images", f"{sub_id}{l + 1}.png"
                    )
                    wtxtp = os.path.join(
                        data_processed_path, "words", "text", f"{sub_id}{l + 1}.txt"
                    )
                    cv.imwrite(
                        wimgp,
                        wbbox_img_inv_bin,
                    )
                    with open(wtxtp, "w") as f:
                        f.write(word)
