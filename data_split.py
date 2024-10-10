import os

# 데이터셋 폴더 경로
data_folder = './data/points'
label_folder = './data/labels'

# 파일 리스트 불러오기 (train 데이터셋 전부)
data_list = os.listdir(data_folder)
data_list = [f[:-4] for f in data_list if f.endswith('.npy')]

label_list = os.listdir(label_folder)
label_list = [f[:-4] for f in label_list if f.endswith('.txt')]

# labels에 있는 데이터는 학습 데이터로, 없는 데이터는 제출용 테스트 데이터로 분리
train_data_list = []
final_test_data_list = []

for file_name in data_list:
    if file_name in label_list:
        train_data_list.append(file_name)
    else:
        final_test_data_list.append(file_name)

# train데이터 중에서 학습, 검증 테스트로 사용할 데이터 분리
# 전체 데이터의 70퍼센트는 학습, 24퍼센트는 검증, 6퍼센트는 최종 테스트 검증에 사용

train_data_idx = int(len(train_data_list) * 0.7)
val_data_idx = int(len(train_data_list) * 0.24)

train_data = train_data_list[:train_data_idx]
val_data = train_data_list[train_data_idx:train_data_idx + val_data_idx]
test_data = train_data_list[train_data_idx + val_data_idx:]

def save_ImageSets(data, file, dir_path):
    # 폴더가 존재하지 않으면 새로 만들기
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # 텍스트 파일 경로 설정 (폴더 안에 파일을 저장)
    file_path = os.path.join(dir_path, file)

    # 리스트 데이터를 텍스트 파일로 저장
    with open(file_path, 'w') as f:
        for item in data:
            f.write(item + '\n')

    print(f"{file_path} 경로에 파일 저장 완료")



# 새 폴더 경로 설정
folder_path = './data/ImageSets'

save_ImageSets(train_data, 'train.txt', folder_path)
save_ImageSets(val_data, 'val.txt', folder_path)
# save_ImageSets(train_data, 'test.txt', folder_path)