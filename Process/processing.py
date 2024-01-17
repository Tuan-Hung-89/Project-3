import os
import pandas as pd
import numpy as np

data_folder = "../Data/Acc/Data1"  # Đường dẫn tới thư mục data1
datatraintest_folder = "../Data/Acc/Datatrain"  # Đường dẫn tới thư mục datatrain
max_data_length = 141


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


# Lặp qua các file trong thư mục data1
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
        indices = np.append(indices, closest_index)  # Thêm closest_index vào indices
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
