
import os
from utils import  *
import shutil
def insert_data():
    """
        thêm những file cat và thư mục cat dog vào thư mục dog
    """
    image_paths = sorted(
    [
        os.path.join("data-classfication/train/train/", file_name)
        for file_name in os.listdir("data-classfication/train/train/")
        if file_name.endswith(".jpg")
    ])

    # tạo thư mục dogs and cats
    try:
        os.mkdir("data-classfication/training")
        os.mkdir("data-classfication/training/dogs")
        os.mkdir("data-classfication/training/cats")
    except:
        print ("Thu muc khong duoc tao")

    i = 0
    for index in range(len(image_paths)):

        path = image_paths[index]

        list_part = path.split("/")
        source = path
        if list_part[-1].startswith("cat"):
            destination = "data-classfication/training/cats/" + list_part[-1]
            shutil.copyfile(source, destination)
        else:
            destination = "data-classfication/training/dogs/" + list_part[-1]
            shutil.copyfile(source, destination)
        i += 1
    if (i == len(image_paths)):
        print("Duyệt thanh công và không có lổi xảy ra")
    else:
        print("Có lổi phát sinh và có: ", abs(len(image_paths) - i))


insert_data()