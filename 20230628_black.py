
import cv2 # opencv-python 라이브러리를 호출
import numpy as np # numpy 라이브러리 호출 -> numpy 라이브러리를 뒤에 코드에서는 np라고 쓰겠다.
import matplotlib.pyplot as plt # matplotlib 시각화 라이브러리
import random
# cv2 : 이미지를 다루는 함수들의 모음 - 이미지 변환 (그레이 스케일로 불러오기, 회전)
# matplotlib.pyplot : 시각화 라이브러리 (그래프)
# Seaborn : 데이터 분포 시각화 라이브러리 

# imread 함수를 이용해서 이미지를 불러오기 -> numpy array 형태로 불러옴
# numpy배열.shape을 통해 이미지의 (세로, 가로, 채널 수)를 확인할 수 있다.
# ./1.jpg: 현재위치에 1.jpg, ../1.jpg: 직전 폴더에 1.jpg, ./image/1.jpg: 현재 위치의 image 폴더 안에 1.jpg
image_ = cv2.imread('./1.jpg') # cv2.imread('이미지의 경로(문자열)')
image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
# 변수명.shape : numpy array (배열)의 차원을 반환

# 흑백 사진으로 변경
#gray = cv2.cvtColor(image_, cv2.COLOR_BGR2GRAY) # cv2.cvtColor(색을 변환시킬 이미지, 어떻게 변화시킬지) : 이미지 색 변환
#cv2.imshow("Gray", gray)                        # cv2.imshow()와 cv2.waitKey()를 같이 써주는게 안정적인 코드
#cv2.waitKey(300)

# 대각선 진행
Trans_image = image_.copy()        # 원본 이미지를 복제해서 수정하기 위한 Trans_image 생성 (복구에 용이하게)
check_sum = np.sum(Trans_image)    # 이미지의 모든 요소들이 0이 되는지 검사하기 위한 변수 (반복문을 탈출하기 위한 조건)
idx = 0                            # 각 check_order 마다 시작점을 정의해주기 위한 변수
box_size = 16                      # 사라지거나 붙일 박스 크기

# 1. 호랑이 이미지를 순차적으로 사라지게 하는 부분
while True:    
    if idx == 0:
        # 원본 이미지를 확인
        cv2.imshow("Work", Trans_image) # cv2.imshow('Figure 창 이름', 픽셀 배열 (이미지))
        cv2.waitKey(10)                 # cv2.waitkey() 
        
        Trans_image[:box_size, :box_size, :] = 0 # 시작점 (0, 0) 부분에 해당하는 픽셀값을 0으로 바꿔준다.
        idx += 1                     # idx 1 증가
        
        cv2.imshow("Work", Trans_image) # cv2.imshow('Figure 창 이름', 픽셀 배열 (이미지))
        cv2.waitKey(10)                 # cv2.waitkey() 
        continue
    
    if idx < (Trans_image.shape[1] // 16):
        # 왼쪽 아래로 내려옴
        if idx % 2 == 1:
            dx = 0
            dy = idx

            while dy >= 0:
                Trans_image[(dx * box_size):((dx + 1) * box_size), (dy * box_size):((dy + 1) * box_size), :] = 0
                dx += 1
                dy -= 1         
                cv2.imshow("Work", Trans_image)
                cv2.waitKey(10)
        # 오른쪽 위로 올라감
        else:
            dx = idx
            dy = 0
            
            while dx >= 0:
                Trans_image[(dx * box_size):((dx + 1) * box_size), (dy * box_size):((dy + 1) * box_size), :] = 0
                dx -= 1
                dy += 1
                cv2.imshow("Work", Trans_image)
                cv2.waitKey(10)
                
    elif (idx >= (Trans_image.shape[1] // 16)) & (idx < (Trans_image.shape[0] // 16)):
       # 왼쪽 아래로 내려옴
        if idx % 2 == 1:
            dx = idx - (Trans_image.shape[1] // 16)
            dy = (Trans_image.shape[1] // 16)

            while dy >= 0:
                Trans_image[(dx * box_size):((dx + 1) * box_size), (dy * box_size):((dy + 1) * box_size), :] = 0
                dx += 1
                dy -= 1         
                cv2.imshow("Work", Trans_image)
                cv2.waitKey(10)
        # 오른쪽 위로 올라감
        else:
            dx = idx
            dy = 0
            
            while dx >= (idx - (Trans_image.shape[1] // 16) + 1):
                Trans_image[(dx * box_size):((dx + 1) * box_size), (dy * box_size):((dy + 1) * box_size), :] = 0
                dx -= 1
                dy += 1
                cv2.imshow("Work", Trans_image)
                cv2.waitKey(10)
                
    else:
        # 왼쪽 아래로 내려옴
        if idx % 2 == 1:
            dx = idx - (Trans_image.shape[1] // 16)
            dy = (Trans_image.shape[1] // 16)

            while dy >= 0:
                Trans_image[(dx * box_size):((dx + 1) * box_size), (dy * box_size):((dy + 1) * box_size), :] = 0
                dx += 1
                dy -= 1         
                cv2.imshow("Work", Trans_image)
                cv2.waitKey(10)
        # 오른쪽 위로 올라감
        else:
            dx = (Trans_image.shape[0] // 16)
            dy = (idx - (Trans_image.shape[0] // 16))
            
            while dx >= (idx - (Trans_image.shape[1] // 16) + 1):
                Trans_image[(dx * box_size):((dx + 1) * box_size), (dy * box_size):((dy + 1) * box_size), :] = 0
                dx -= 1
                dy += 1
                cv2.imshow("Work", Trans_image)
                cv2.waitKey(10)
               
    idx += 1
    check_sum = np.sum(Trans_image)
    
    if check_sum == 0:
        break

# 2. 첫 번째 이미지를 심장 형태로 채우는 부분
image2_ = cv2.imread('./2.jpg')                     # cv2.imread('이미지의 경로(문자열)'), 데이터 타입을 따로 설정하지 않으면 'uint8'
image2_ = cv2.cvtColor(image2_, cv2.COLOR_BGR2RGB)  # cv2가 이미지를 불러올 때 (Blue, Green, Red) 형태로 불러옴 -> RGB 형태로 변환

# 색은 0 ~ 255 픽셀 값을 가지고 있고, 0은 검은색, 1은 하얀색
# np.zeros((차원), dtype = np.uint8) : 해당 차원의 모양을 가지고 있는 배열을 생성하는데 값을 0으로 채움, 원하는 배열을 생성할 때 (초기화할 때)
# np.ones((차원)) : 값을 1로 채움
# np.empty((차원)) : 비어있는 numpy array 생성
# 리스트를 numpy array로 변환
# dtype = np.uint8: cv2로 데이터를 불러올때 데이터 타입과 통일시켜주기 위해
Trans_image = np.zeros((image2_.shape[0], image2_.shape[1], 3), dtype = np.uint8)

cv2.imshow("Work", Trans_image) # cv2.imshow('Figure 창 이름', 픽셀 배열 (이미지))
cv2.waitKey(10)                 # cv2.waitkey() 

# 2-1. 대동맥 부분
for idx_row in range(0, 1):
    for idx_col in range(20, 25):       
        select_row = random.randint(3, 32) # 랜덤하게 들어갈 호랑이 이미지의 좌표를 추출
        select_col = random.randint(6, 19) # 랜덤하게 들어갈 호랑이 이미지의 좌표를 추출
        
        row_start = ((idx_row) * box_size)
        row_end = ((idx_row + 1) * box_size)
        
        col_start = ((idx_col) * box_size)
        col_end = ((idx_col + 1) * box_size)
        
        inserted_image = image_[((select_row - 1) * (box_size * 2)):(select_row * (box_size * 2)), ((select_col - 1) * (box_size * 2)):(select_col * (box_size * 2)), :]
        inserted_image = cv2.resize(inserted_image, dsize = (16, 16))
        Trans_image[row_start:row_end, col_start:col_end, :] = inserted_image.copy()
        cv2.imshow("Work", Trans_image)
        cv2.waitKey(10)
    
    for idx_col in range(28, 35):       
        select_row = random.randint(3, 32)
        select_col = random.randint(6, 19)
        
        row_start = ((idx_row) * box_size)
        row_end = ((idx_row + 1) * box_size)
        
        col_start = ((idx_col) * box_size)
        col_end = ((idx_col + 1) * box_size)
        
        inserted_image = image_[((select_row - 1) * (box_size * 2)):(select_row * (box_size * 2)), ((select_col - 1) * (box_size * 2)):(select_col * (box_size * 2)), :]
        inserted_image = cv2.resize(inserted_image, dsize = (16, 16))
        Trans_image[row_start:row_end, col_start:col_end, :] = inserted_image.copy()
        cv2.imshow("Work", Trans_image)
        cv2.waitKey(10)

for idx_row in range(1, 2):
    for idx_col in range(21, 26):       
        select_row = random.randint(3, 32)
        select_col = random.randint(6, 19)
        
        row_start = ((idx_row) * box_size)
        row_end = ((idx_row + 1) * box_size)
        
        col_start = ((idx_col) * box_size)
        col_end = ((idx_col + 1) * box_size)
        
        inserted_image = image_[((select_row - 1) * (box_size * 2)):(select_row * (box_size * 2)), ((select_col - 1) * (box_size * 2)):(select_col * (box_size * 2)), :]
        inserted_image = cv2.resize(inserted_image, dsize = (16, 16))
        Trans_image[row_start:row_end, col_start:col_end, :] = inserted_image.copy()
        cv2.imshow("Work", Trans_image)
        cv2.waitKey(10)
    
    for idx_col in range(29, 34):       
        select_row = random.randint(3, 32)
        select_col = random.randint(6, 19)
        
        row_start = ((idx_row) * box_size)
        row_end = ((idx_row + 1) * box_size)
        
        col_start = ((idx_col) * box_size)
        col_end = ((idx_col + 1) * box_size)
        
        inserted_image = image_[((select_row - 1) * (box_size * 2)):(select_row * (box_size * 2)), ((select_col - 1) * (box_size * 2)):(select_col * (box_size * 2)), :]
        inserted_image = cv2.resize(inserted_image, dsize = (16, 16))
        Trans_image[row_start:row_end, col_start:col_end, :] = inserted_image.copy()
        cv2.imshow("Work", Trans_image)
        cv2.waitKey(10)

for idx_row in range(2, 3):
    for idx_col in range(20, 33):       
        select_row = random.randint(3, 32)
        select_col = random.randint(6, 19)
        
        row_start = ((idx_row) * box_size)
        row_end = ((idx_row + 1) * box_size)
        
        col_start = ((idx_col) * box_size)
        col_end = ((idx_col + 1) * box_size)
        
        inserted_image = image_[((select_row - 1) * (box_size * 2)):(select_row * (box_size * 2)), ((select_col - 1) * (box_size * 2)):(select_col * (box_size * 2)), :]
        inserted_image = cv2.resize(inserted_image, dsize = (16, 16))
        Trans_image[row_start:row_end, col_start:col_end, :] = inserted_image.copy()
        cv2.imshow("Work", Trans_image)
        cv2.waitKey(10)

for idx_row in range(3, 5):
    for idx_col in range(21, 32):       
        select_row = random.randint(3, 32)
        select_col = random.randint(6, 19)
        
        row_start = ((idx_row) * box_size)
        row_end = ((idx_row + 1) * box_size)
        
        col_start = ((idx_col) * box_size)
        col_end = ((idx_col + 1) * box_size)
        
        inserted_image = image_[((select_row - 1) * (box_size * 2)):(select_row * (box_size * 2)), ((select_col - 1) * (box_size * 2)):(select_col * (box_size * 2)), :]
        inserted_image = cv2.resize(inserted_image, dsize = (16, 16))
        Trans_image[row_start:row_end, col_start:col_end, :] = inserted_image.copy()
        cv2.imshow("Work", Trans_image)
        cv2.waitKey(10)

for idx_row in range(5, 7):
    for idx_col in range(24, 30):       
        select_row = random.randint(3, 32)
        select_col = random.randint(6, 19)
        
        row_start = ((idx_row) * box_size)
        row_end = ((idx_row + 1) * box_size)
        
        col_start = ((idx_col) * box_size)
        col_end = ((idx_col + 1) * box_size)
        
        inserted_image = image_[((select_row - 1) * (box_size * 2)):(select_row * (box_size * 2)), ((select_col - 1) * (box_size * 2)):(select_col * (box_size * 2)), :]
        inserted_image = cv2.resize(inserted_image, dsize = (16, 16))
        Trans_image[row_start:row_end, col_start:col_end, :] = inserted_image.copy()
        cv2.imshow("Work", Trans_image)
        cv2.waitKey(10)


# 2-2. 심장 부분
for idx_row in range(7, 20):
    for idx_col in range(21, 43):       
        select_row = random.randint(3, 32)
        select_col = random.randint(6, 19)
        
        row_start = ((idx_row) * box_size)
        row_end = ((idx_row + 1) * box_size)
        
        col_start = ((idx_col) * box_size)
        col_end = ((idx_col + 1) * box_size)
        
        inserted_image = image_[((select_row - 1) * (box_size * 2)):(select_row * (box_size * 2)), ((select_col - 1) * (box_size * 2)):(select_col * (box_size * 2)), :]
        inserted_image = cv2.resize(inserted_image, dsize = (16, 16))
        Trans_image[row_start:row_end, col_start:col_end, :] = inserted_image.copy()
        cv2.imshow("Work", Trans_image)
        cv2.waitKey(10)
    
for idx_row in range(20, 22):
    for idx_col in range(20, 45):
        select_row = random.randint(3, 32)
        select_col = random.randint(6, 19)
        
        row_start = ((idx_row) * box_size)
        row_end = ((idx_row + 1) * box_size)
        
        col_start = ((idx_col) * box_size)
        col_end = ((idx_col + 1) * box_size)
        
        inserted_image = image_[((select_row - 1) * (box_size * 2)):(select_row * (box_size * 2)), ((select_col - 1) * (box_size * 2)):(select_col * (box_size * 2)), :]
        inserted_image = cv2.resize(inserted_image, dsize = (16, 16))
        Trans_image[row_start:row_end, col_start:col_end, :] = inserted_image.copy()
        
        cv2.imshow("Work", Trans_image)
        cv2.waitKey(10)

for idx_row in range(22, 25):
    for idx_col in range(21, 46):
        select_row = random.randint(3, 32)
        select_col = random.randint(6, 19)
        
        row_start = ((idx_row) * box_size)
        row_end = ((idx_row + 1) * box_size)
        
        col_start = ((idx_col) * box_size)
        col_end = ((idx_col + 1) * box_size)
        
        inserted_image = image_[((select_row - 1) * (box_size * 2)):(select_row * (box_size * 2)), ((select_col - 1) * (box_size * 2)):(select_col * (box_size * 2)), :]
        inserted_image = cv2.resize(inserted_image, dsize = (16, 16))
        Trans_image[row_start:row_end, col_start:col_end, :] = inserted_image.copy()
        
        cv2.imshow("Work", Trans_image)
        cv2.waitKey(10)

for idx_row in range(25, 27):
    for idx_col in range(22, 46):
        select_row = random.randint(3, 32)
        select_col = random.randint(6, 19)
        
        row_start = ((idx_row) * box_size)
        row_end = ((idx_row + 1) * box_size)
        
        col_start = ((idx_col) * box_size)
        col_end = ((idx_col + 1) * box_size)
        
        inserted_image = image_[((select_row - 1) * (box_size * 2)):(select_row * (box_size * 2)), ((select_col - 1) * (box_size * 2)):(select_col * (box_size * 2)), :]
        inserted_image = cv2.resize(inserted_image, dsize = (16, 16))
        Trans_image[row_start:row_end, col_start:col_end, :] = inserted_image.copy()
        
        cv2.imshow("Work", Trans_image)
        cv2.waitKey(10)

for idx_row in range(27, 28):
    for idx_col in range(23, 46):
        select_row = random.randint(3, 32)
        select_col = random.randint(6, 19)
        
        row_start = ((idx_row) * box_size)
        row_end = ((idx_row + 1) * box_size)
        
        col_start = ((idx_col) * box_size)
        col_end = ((idx_col + 1) * box_size)
        
        inserted_image = image_[((select_row - 1) * (box_size * 2)):(select_row * (box_size * 2)), ((select_col - 1) * (box_size * 2)):(select_col * (box_size * 2)), :]
        inserted_image = cv2.resize(inserted_image, dsize = (16, 16))
        Trans_image[row_start:row_end, col_start:col_end, :] = inserted_image.copy()
        
        cv2.imshow("Work", Trans_image)
        cv2.waitKey(10)

for idx_row in range(28, 29):
    for idx_col in range(24, 46):
        select_row = random.randint(3, 32)
        select_col = random.randint(6, 19)
        
        row_start = ((idx_row) * box_size)
        row_end = ((idx_row + 1) * box_size)
        
        col_start = ((idx_col) * box_size)
        col_end = ((idx_col + 1) * box_size)
        
        inserted_image = image_[((select_row - 1) * (box_size * 2)):(select_row * (box_size * 2)), ((select_col - 1) * (box_size * 2)):(select_col * (box_size * 2)), :]
        inserted_image = cv2.resize(inserted_image, dsize = (16, 16))
        Trans_image[row_start:row_end, col_start:col_end, :] = inserted_image.copy()
        
        cv2.imshow("Work", Trans_image)
        cv2.waitKey(10)
        
for idx_row in range(29, 31):
    for idx_col in range(25, 46):
        select_row = random.randint(3, 32)
        select_col = random.randint(6, 19)
        
        row_start = ((idx_row) * box_size)
        row_end = ((idx_row + 1) * box_size)
        
        col_start = ((idx_col) * box_size)
        col_end = ((idx_col + 1) * box_size)
        
        inserted_image = image_[((select_row - 1) * (box_size * 2)):(select_row * (box_size * 2)), ((select_col - 1) * (box_size * 2)):(select_col * (box_size * 2)), :]
        inserted_image = cv2.resize(inserted_image, dsize = (16, 16))
        Trans_image[row_start:row_end, col_start:col_end, :] = inserted_image.copy()
        
        cv2.imshow("Work", Trans_image)
        cv2.waitKey(10)
        
for idx_row in range(31, 33):
    for idx_col in range(27, 45):
        select_row = random.randint(3, 32)
        select_col = random.randint(6, 19)
        
        row_start = ((idx_row) * box_size)
        row_end = ((idx_row + 1) * box_size)
        
        col_start = ((idx_col) * box_size)
        col_end = ((idx_col + 1) * box_size)
        
        inserted_image = image_[((select_row - 1) * (box_size * 2)):(select_row * (box_size * 2)), ((select_col - 1) * (box_size * 2)):(select_col * (box_size * 2)), :]
        inserted_image = cv2.resize(inserted_image, dsize = (16, 16))
        Trans_image[row_start:row_end, col_start:col_end, :] = inserted_image.copy()
        
        cv2.imshow("Work", Trans_image)
        cv2.waitKey(10)

for idx_row in range(33, 34):
    for idx_col in range(30, 45):
        select_row = random.randint(3, 32)
        select_col = random.randint(6, 19)
        
        row_start = ((idx_row) * box_size)
        row_end = ((idx_row + 1) * box_size)
        
        col_start = ((idx_col) * box_size)
        col_end = ((idx_col + 1) * box_size)
        
        inserted_image = image_[((select_row - 1) * (box_size * 2)):(select_row * (box_size * 2)), ((select_col - 1) * (box_size * 2)):(select_col * (box_size * 2)), :]
        inserted_image = cv2.resize(inserted_image, dsize = (16, 16))
        Trans_image[row_start:row_end, col_start:col_end, :] = inserted_image.copy()
        
        cv2.imshow("Work", Trans_image)
        cv2.waitKey(10)

for idx_row in range(34, 35):
    for idx_col in range(32, 45):
        select_row = random.randint(3, 32)
        select_col = random.randint(6, 19)
        
        row_start = ((idx_row) * box_size)
        row_end = ((idx_row + 1) * box_size)
        
        col_start = ((idx_col) * box_size)
        col_end = ((idx_col + 1) * box_size)
        
        inserted_image = image_[((select_row - 1) * (box_size * 2)):(select_row * (box_size * 2)), ((select_col - 1) * (box_size * 2)):(select_col * (box_size * 2)), :]
        inserted_image = cv2.resize(inserted_image, dsize = (16, 16))
        Trans_image[row_start:row_end, col_start:col_end, :] = inserted_image.copy()
        
        cv2.imshow("Work", Trans_image)
        cv2.waitKey(10)

for idx_row in range(35, 36):
    for idx_col in range(34, 45):
        select_row = random.randint(3, 32)
        select_col = random.randint(6, 19)
        
        row_start = ((idx_row) * box_size)
        row_end = ((idx_row + 1) * box_size)
        
        col_start = ((idx_col) * box_size)
        col_end = ((idx_col + 1) * box_size)
        
        inserted_image = image_[((select_row - 1) * (box_size * 2)):(select_row * (box_size * 2)), ((select_col - 1) * (box_size * 2)):(select_col * (box_size * 2)), :]
        inserted_image = cv2.resize(inserted_image, dsize = (16, 16))
        Trans_image[row_start:row_end, col_start:col_end, :] = inserted_image.copy()
        
        cv2.imshow("Work", Trans_image)
        cv2.waitKey(10)

cv2.imshow("Work", Trans_image)       
cv2.waitKey()