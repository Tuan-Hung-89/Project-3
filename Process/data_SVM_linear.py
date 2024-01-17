import os
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

results = []


def truncate_or_pad_data(data, length):
    if len(data) < length:
        # Đệm dữ liệu bằng số 0 ở cuối
        padded_data = np.pad(data, (0, length - len(data)), mode="constant")
        return padded_data
    elif len(data) > length:
        # Cắt bớt dữ liệu theo độ dài mong muốn
        truncated_data = data[:length]
        return truncated_data
    else:
        return data


def data_processing_train(max_data_length):
    data_folder = "../Data/Acc/Data1"  # Đường dẫn tới thư mục data1
    datatraintest_folder = "../Data/Acc/Datatrain"  # Đường dẫn tới thư mục datatrain
    for file_name in os.listdir(data_folder):
        # Đọc file CSV
        file_path = os.path.join(data_folder, file_name)
        dataset = pd.read_csv(file_path)

        # Tiến hành xử lý tương tự như đoạn mã ban đầu
        N = dataset["Ax"].size
        t_step = 0.01
        t = np.arange(0, (N - 0.5) * t_step, t_step)
        Ax_y = dataset["Ax"]
        Ay_y = dataset["Ay"]
        Az_y = dataset["Az"]

        Ax = np.fft.fft(Ax_y)
        Ax_plot = Ax[0 : int(t.size / 2 + 1)]

        Ay = np.fft.fft(Ay_y)
        Ay_plot = Ay[0 : int(t.size / 2 + 1)]

        Az = np.fft.fft(Az_y)
        Az_plot = Az[0 : int(t.size / 2 + 1)]

        t_plot = np.linspace(0, 100, Ax_plot.size)

        frequency = 50  # Tần số cần tìm

        indices = np.where(t_plot == frequency)[0]

        if len(indices) == 0:
            # Tìm vị trí của phần tử gần nhất với hoặc bằng frequency trong mảng t_plot
            closest_index = np.argmin(np.abs(t_plot - frequency))
            indices = np.append(
                indices, closest_index
            )  # Thêm closest_index vào indices
            print("Vị trí của phần tử gần 50:", indices)

        else:
            # In ra vị trí của các phần tử có tần số bằng frequency
            print("Vị trí của phần tử có tần số 50:", indices)

        # Thiết kế bộ lọc thông thấp
        cutoff_frequency = indices[0]  # Tần số cắt (cutoff frequency)
        # Áp dụng bộ lọc
        filtered_Ax = Ax_plot[0:cutoff_frequency]
        filtered_Ay = Ay_plot[0:cutoff_frequency]
        filtered_Az = Az_plot[0:cutoff_frequency]

        filtered_Ax = truncate_or_pad_data(filtered_Ax, max_data_length)
        filtered_Ay = truncate_or_pad_data(filtered_Ay, max_data_length)
        filtered_Az = truncate_or_pad_data(filtered_Az, max_data_length)
        # Tạo DataFrame mới từ dữ liệu đã lọc
        df = pd.DataFrame(
            {
                "Ax": np.abs(filtered_Ax.astype(complex)),
                "Ay": np.abs(filtered_Ay.astype(complex)),
                "Az": np.abs(filtered_Az.astype(complex)),
            }
        )

        # Xác định output_file dựa trên tên của file CSV hiện tại
        output_file = os.path.join(
            datatraintest_folder, file_name
        )  # Giữ nguyên tên file đầu vào
        # Lưu DataFrame vào file CSV
        try:
            df.to_csv(output_file, index=False)
            print(f"File {output_file} đã được lưu.")
        except Exception as e:
            print(f"Lỗi khi lưu file {output_file}: {e}")
            exit()


def data_processing_test(max_data_length):
    data_folder = "../Data/Acc/Data2"  # Đường dẫn tới thư mục data1
    datatraintest_folder = "../Data/Acc/Datatest"  # Đường dẫn tới thư mục datatrain
    for file_name in os.listdir(data_folder):
        # Đọc file CSV
        file_path = os.path.join(data_folder, file_name)
        dataset = pd.read_csv(file_path)

        # Tiến hành xử lý tương tự như đoạn mã ban đầu
        N = dataset["Ax"].size
        t_step = 0.01
        t = np.arange(0, (N - 0.5) * t_step, t_step)
        Ax_y = dataset["Ax"]
        Ay_y = dataset["Ay"]
        Az_y = dataset["Az"]

        Ax = np.fft.fft(Ax_y)
        Ax_plot = Ax[0 : int(t.size / 2 + 1)]

        Ay = np.fft.fft(Ay_y)
        Ay_plot = Ay[0 : int(t.size / 2 + 1)]

        Az = np.fft.fft(Az_y)
        Az_plot = Az[0 : int(t.size / 2 + 1)]

        t_plot = np.linspace(0, 100, Ax_plot.size)

        frequency = 50  # Tần số cần tìm

        indices = np.where(t_plot == frequency)[0]

        if len(indices) == 0:
            # Tìm vị trí của phần tử gần nhất với hoặc bằng frequency trong mảng t_plot
            closest_index = np.argmin(np.abs(t_plot - frequency))
            indices = np.append(
                indices, closest_index
            )  # Thêm closest_index vào indices
            print("Vị trí của phần tử gần 50:", indices)

        else:
            # In ra vị trí của các phần tử có tần số bằng frequency
            print("Vị trí của phần tử có tần số 50:", indices)

        # Thiết kế bộ lọc thông thấp
        cutoff_frequency = indices[0]  # Tần số cắt (cutoff frequency)
        # Áp dụng bộ lọc
        filtered_Ax = Ax_plot[0:cutoff_frequency]
        filtered_Ay = Ay_plot[0:cutoff_frequency]
        filtered_Az = Az_plot[0:cutoff_frequency]

        filtered_Ax = truncate_or_pad_data(filtered_Ax, max_data_length)
        filtered_Ay = truncate_or_pad_data(filtered_Ay, max_data_length)
        filtered_Az = truncate_or_pad_data(filtered_Az, max_data_length)
        # Tạo DataFrame mới từ dữ liệu đã lọc
        df = pd.DataFrame(
            {
                "Ax": np.abs(filtered_Ax.astype(complex)),
                "Ay": np.abs(filtered_Ay.astype(complex)),
                "Az": np.abs(filtered_Az.astype(complex)),
            }
        )

        # Xác định output_file dựa trên tên của file CSV hiện tại
        output_file = os.path.join(
            datatraintest_folder, file_name
        )  # Giữ nguyên tên file đầu vào
        # Lưu DataFrame vào file CSV
        try:
            df.to_csv(output_file, index=False)
            print(f"File {output_file} đã được lưu.")
        except Exception as e:
            print(f"Lỗi khi lưu file {output_file}: {e}")
            exit()


def train_test():
    # Đường dẫn đến thư mục chứa dữ liệu
    data_processing_dir = "../Data/Acc/Datatrain"

    # Xác định các đặc trưng (features) và nhãn (labels)
    features = ["Ax", "Ay", "Az"]

    # Chuẩn bị dữ liệu huấn luyện
    train_data = []
    train_labels = []

    # Đọc dữ liệu từ thư mục Data_processing
    for file_name in os.listdir(data_processing_dir):
        if file_name.endswith("_0.csv"):
            label = 0  # normal
        elif file_name.endswith("_1.csv"):
            label = 1  # game
        else:
            continue

        file_path = os.path.join(data_processing_dir, file_name)
        df = pd.read_csv(file_path)
        flattened_data = df[
            features
        ].values.flatten()  # Chuyển đổi dữ liệu thành 2 chiều
        train_data.append(flattened_data)
        train_labels.append(label)
    # Xây dựng mô hình SVM
    svm_model = SVC(kernel="linear")
    svm_model.fit(train_data, train_labels)

    test_path = "../Data/Acc/Datatest"

    test_data = []
    test_filenames = []
    test_labels = []

    # Duyệt qua các tệp tin trong thư mục test_path
    for filename in os.listdir(test_path):
        # Kiểm tra nếu tệp tin có đuôi .csv
        if filename.endswith(".csv"):
            file_path = os.path.join(test_path, filename)
            df = pd.read_csv(file_path)

            # Lấy dữ liệu từ 3 cột Ax, Ay, Az
            features = df[["Ax", "Ay", "Az"]].values
            test_data.append(features)  # Thêm dữ liệu vào test_data
            test_filenames.append(filename)  # Thêm tên tệp tin vào test_filenames
            test_labels.append(
                0 if filename.endswith("_0.csv") else 1
            )  # Thêm nhãn vào test_labels (0: nữ, 1: nam)

    test_data = np.array(test_data).reshape(
        len(test_data), -1
    )  # Chuyển đổi test_data thành mảng numpy 2D

    predictions = svm_model.predict(
        test_data
    )  # Dự đoán giới tính trên dữ liệu kiểm tra

    # In kết quả dự đoán và tên tệp tin tương ứng
    for filename, label, prediction in zip(test_filenames, test_labels, predictions):
        print("File:", filename)
        print("Dự đoán kiểu đi", prediction)
        print("--------------------")

    accuracy = accuracy_score(
        test_labels, predictions
    )  # Tính độ chính xác bằng cách so sánh nhãn thực tế và nhãn dự đoán
    print(
        "Độ chính xác của mô hình SVM trên dữ liệu kiểm tra: {:.2f}%".format(
            accuracy * 100
        )
    )
    temp = accuracy
    results.append(temp * 100)


for i in range(0, 110):
    data_processing_test(i + 30)
    data_processing_train(i + 30)
    train_test()


df = pd.DataFrame({"predictions": np.abs(results)})
df.to_csv("../results/results_linear.csv", index=False)
