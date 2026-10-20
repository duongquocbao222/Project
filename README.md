# Scientific Research: Tối ưu hóa Fluid Antenna Multiple Access bằng Deep Learning

**Nghiên cứu hệ thống Fluid Antenna Multiple Access (FAMA), mô hình tương quan khối và ứng dụng Deep Learning trong tối ưu hóa phân chia cổng anten.**

## 1. Giới thiệu dự án

Repository này lưu trữ mã nguồn MATLAB phục vụ nghiên cứu và mô phỏng hệ thống **Fluid Antenna System (FAS)** và **Fluid Antenna Multiple Access (FAMA)**, tập trung vào việc tối ưu hóa hiệu năng truyền thông không dây thông qua mô hình hóa tương quan kênh truyền và ứng dụng học sâu.

Trong hệ thống anten thông thường, vị trí anten thường cố định. Với Fluid Antenna System, một anten có thể lựa chọn giữa nhiều vị trí hoặc cổng anten (*ports*) trong một vùng không gian giới hạn. Việc lựa chọn cổng phù hợp giúp hệ thống tận dụng sự biến thiên của kênh truyền để cải thiện chất lượng thu tín hiệu.

Khi nhiều người dùng cùng chia sẻ tài nguyên truyền thông, nhiễu đồng kênh và tương quan giữa các cổng anten có thể ảnh hưởng đáng kể đến hiệu năng hệ thống. Vì vậy, việc mô hình hóa tương quan kênh và tìm cách phân chia các cổng anten thành những khối phù hợp là một bài toán quan trọng.

Dự án triển khai và khảo sát ba hướng tiếp cận chính:

- **Jakes correlation model:** Mô hình tương quan kênh truyền được sử dụng làm cơ sở mô phỏng và đánh giá.
- **Block-Correlation (BC):** Phương pháp xấp xỉ cấu trúc tương quan bằng ma trận dạng khối, hỗ trợ tính toán xác suất gián đoạn truyền dẫn.
- **Deep Learning (DL):** Sử dụng mạng nơ-ron để dự đoán phương án phân chia cổng anten, hướng đến giảm chi phí tính toán của quá trình tối ưu.

Mục tiêu là đánh giá khả năng sử dụng Deep Learning để dự đoán cấu hình phân chia cổng anten, đồng thời so sánh hiệu năng và thời gian tính toán với phương pháp dựa trên mô hình tương quan khối.

## 2. Mục tiêu nghiên cứu

Các mục tiêu chính của dự án gồm:

1. Tìm hiểu và mô phỏng hệ thống FAS/FAMA trong môi trường fading Rayleigh có tương quan.
2. Xây dựng mô hình tương quan kênh truyền dựa trên Jakes và phương pháp xấp xỉ Block-Correlation.
3. Tính toán xác suất gián đoạn truyền dẫn (*Outage Probability – OP*) theo các tham số của hệ thống.
4. Xây dựng tập dữ liệu từ quá trình tìm kiếm cấu hình phân chia cổng anten có OP thấp.
5. Huấn luyện mạng Deep Learning để dự đoán cấu hình phân chia cổng anten từ các tham số đầu vào.
6. So sánh phương pháp dựa trên Deep Learning với phương pháp Block-Correlation và kết quả mô phỏng theo mô hình Jakes.
7. Khảo sát thời gian tính toán và khả năng ứng dụng của mô hình học sâu trong quá trình tối ưu hóa.

## 3. Cơ sở lý thuyết

### 3.1. Fluid Antenna System (FAS)

Fluid Antenna System là hệ thống anten cho phép lựa chọn vị trí hoặc cổng anten phù hợp trong một vùng không gian nhất định.

Các tham số quan trọng trong dự án gồm:

| Ký hiệu | Ý nghĩa |
|---|---|
| \(W\) | Chiều dài vùng anten, thường biểu diễn theo bước sóng |
| \(N\) | Tổng số cổng anten |
| \(d\) | Khoảng cách giữa hai cổng anten liên tiếp trong mô hình bố trí đều |
| \(U\) | Số người dùng đồng thời |
| \(\gamma\) | Ngưỡng SIR hoặc SNR, tùy trường hợp tính toán |
| \(L\) | Vector biểu diễn số cổng trong từng khối tương quan |
| \(B\) | Số khối trong phương án phân chia khi sử dụng cấu hình có \(B\) khối |

Với \(N\) cổng được bố trí đều trên vùng anten có chiều dài \(W\), mã nguồn sử dụng:

\[
d=\frac{W}{N-1}
\]

Công thức này giả định \(N>1\).

### 3.2. Fluid Antenna Multiple Access (FAMA)

FAMA mở rộng ý tưởng FAS sang trường hợp nhiều người dùng cùng truy cập hệ thống.

Khi nhiều người dùng truyền tín hiệu đồng thời, tín hiệu mong muốn có thể bị ảnh hưởng bởi nhiễu từ những người dùng khác. Hệ thống có thể lựa chọn cổng anten để tận dụng các trạng thái kênh truyền thuận lợi hơn.

Trong mã nguồn, với trường hợp nhiều người dùng, chất lượng tín hiệu được đánh giá thông qua tỷ số tín hiệu trên nhiễu (SIR). Với trường hợp một người dùng, hàm tính toán sử dụng ngưỡng SNR.

### 3.3. Rayleigh fading và mô hình tương quan Jakes

Rayleigh fading được sử dụng để mô tả sự biến thiên ngẫu nhiên của kênh truyền trong môi trường đa đường khi không xét thành phần đường truyền trực tiếp trội theo mô hình Rayleigh cơ bản.

Mô hình tương quan Jakes sử dụng hàm Bessel loại một, bậc không:

\[
R_{ij}=J_0\left(2\pi\frac{|i-j|W}{N-1}\right)
\]

Trong đó \(i,j\) là chỉ số các cổng anten.

Trong MATLAB, ma trận tương quan được xây dựng bằng:

```matlab
Sigma = toeplitz( ...
    besselj(0, 2*pi*(0:N-1)*W/(N-1)) ...
);
```

Ma trận `Sigma` mô tả tương quan giữa các cổng anten. Ma trận này được sử dụng trong quá trình mô phỏng kênh truyền và đánh giá hiệu năng hệ thống.

### 3.4. Block-Correlation (BC)

Block-Correlation là phương pháp xấp xỉ cấu trúc tương quan của hệ thống bằng các khối, giúp đơn giản hóa việc tính toán xác suất gián đoạn.

Vector phân chia cổng anten được biểu diễn dưới dạng:

\[
L=[L_1,L_2,\ldots,L_B]
\]

với điều kiện tổng số cổng được phân chia phải bằng tổng số cổng anten:

\[
\sum_{b=1}^{B}L_b=N
\]

Mỗi phần tử \(L_b\) biểu diễn số cổng thuộc khối thứ \(b\).

Trong dự án, `BlockCorrelation.m` xác định phương án phân chia dựa trên số lượng cổng, phổ trị riêng và tham số tương quan. Hàm `CalcOutage.m` sau đó tính OP tương ứng với phương án phân chia.

### 3.5. Deep Learning trong tối ưu hóa phân chia cổng

Dự án sử dụng mạng nơ-ron truyền thẳng (*feedforward neural network*) để học mối quan hệ giữa các tham số hệ thống và cấu hình phân chia cổng anten.

Các đầu vào được sử dụng trong mã tạo dữ liệu và huấn luyện gồm:

\[
X=[W,U,\gamma,N]
\]

Đầu ra của mạng gồm \(B=12\) giá trị, tương ứng với tỷ lệ phân chia cổng anten giữa các khối.

Trong chương trình `train_DL_softmax.m`, các đầu ra mục tiêu được chuẩn hóa theo:

\[
Y_{\text{normalized}}=\frac{Y_{\text{raw}}}{N}
\]

Mạng sử dụng các lớp fully connected, batch normalization, ReLU và Softmax. Softmax giúp tổng các giá trị đầu ra bằng 1, phù hợp với cách biểu diễn tỷ lệ phân chia.

Sau khi dự đoán, các tỷ lệ được chuyển thành số lượng cổng nguyên và cần được hiệu chỉnh để bảo đảm tổng số cổng bằng \(N\).

**Lưu ý:** Mạng Deep Learning dự đoán cấu hình dựa trên dữ liệu đã huấn luyện. Kết quả dự đoán cần được đánh giá bằng OP và so sánh với các phương pháp tham chiếu; không mặc định rằng mọi dự đoán đều là nghiệm tối ưu toàn cục.

## 4. Phương pháp nghiên cứu

Quy trình nghiên cứu được chia thành các giai đoạn sau.

### Giai đoạn 1: Mô hình hóa hệ thống

- Xác định các tham số hệ thống gồm \(W\), \(N\), \(U\) và ngưỡng \(\gamma\).
- Xây dựng ma trận tương quan theo mô hình Jakes.
- Xác định hệ số tương quan sử dụng trong xấp xỉ Block-Correlation.

### Giai đoạn 2: Xây dựng mô hình tính OP

Sử dụng các hàm trong thư mục `Core/` để tính xác suất gián đoạn theo mô hình lý thuyết và mô phỏng Monte Carlo.

### Giai đoạn 3: Tạo tập dữ liệu

Sinh các bộ tham số ngẫu nhiên, tạo những phương án phân chia cổng anten thỏa mãn điều kiện tổng số cổng bằng \(N\), sau đó đánh giá OP để lựa chọn cấu hình có OP thấp nhất trong tập phương án được xét.

### Giai đoạn 4: Huấn luyện mạng Deep Learning

- Đọc tập dữ liệu đã sinh.
- Chuẩn hóa đầu ra theo tổng số cổng anten.
- Xáo trộn dữ liệu và chia thành tập huấn luyện, tập kiểm tra.
- Huấn luyện mạng nơ-ron và lưu mô hình.

### Giai đoạn 5: Đánh giá và so sánh

So sánh cấu hình dự đoán bởi mạng Deep Learning với cấu hình Block-Correlation và kết quả mô phỏng theo mô hình Jakes.

Các tiêu chí đánh giá gồm:

- Xác suất gián đoạn OP.
- Mức độ phù hợp của cấu hình phân chia cổng.
- Thời gian tính toán.
- Độ ổn định và tính nhất quán của kết quả mô phỏng.

## 5. Cấu trúc thư mục

Repository hiện tổ chức mã nguồn trong thư mục `NCKH_FAS_BCvsML-main/`.

```text
Scientific_Research/
└── NCKH_FAS_BCvsML-main/
    ├── Core/
    │   ├── ApproxOutage.m
    │   ├── BlockCorrelation.m
    │   ├── CalcOutage.m
    │   ├── DeltaOutage.m
    │   ├── SigmaIso3D.m
    │   ├── SimOutage.m
    │   └── SimOutage_Blocks.m
    │
    ├── Figure1m
    ├── Figure2.m
    ├── GenerateAllL.m
    ├── dataset_generate.m
    ├── dataset_final.mat
    ├── train_DL_softmax.m
    ├── trained_net_softmax.mat
    ├── hinh3.m
    ├── hinh4.m
    ├── hinh5.m
    ├── sosanhComplexity.m
    ├── sosanhOP_SIIR.m
    ├── sosanhOP_fix.m
    ├── sosanhComplexity.m
    ├── fig_so_sanh_complexity.png
    └── so_sanh_Pout.png
```

### 5.1. Thư mục `Core/`

Chứa các hàm tính toán cốt lõi:

| Tệp | Chức năng |
|---|---|
| `CalcOutage.m` | Tính xác suất gián đoạn bằng biểu thức lý thuyết và phương pháp tích phân số hoặc quadrature. |
| `SimOutage.m` | Mô phỏng OP bằng Monte Carlo với ma trận tương quan kênh truyền. |
| `BlockCorrelation.m` | Xây dựng cấu hình phân chia cổng anten theo phương pháp tương quan khối. |
| `SimOutage_Blocks.m` | Mô phỏng OP trong mô hình sử dụng các khối tương quan. |
| `ApproxOutage.m` | Tính toán hoặc xấp xỉ OP theo mô hình được triển khai trong hàm. |
| `DeltaOutage.m` | Đánh giá đại lượng chênh lệch OP theo các tham số đầu vào. |
| `SigmaIso3D.m` | Hàm hỗ trợ liên quan đến ma trận tương quan trong mô hình không gian được triển khai. |

### 5.2. Các chương trình tạo dữ liệu và huấn luyện

| Tệp | Chức năng |
|---|---|
| `GenerateAllL.m` | Sinh các vector phân chia số nguyên có tổng bằng \(N\). |
| `dataset_generate.m` | Sinh tập dữ liệu từ các tham số hệ thống và quá trình tìm kiếm cấu hình có OP thấp. |
| `train_DL_softmax.m` | Chuẩn hóa dữ liệu, xây dựng và huấn luyện mạng Deep Learning. |
| `dataset_final.mat` | Tệp dữ liệu huấn luyện được chương trình sử dụng. |
| `trained_net_softmax.mat` | Tệp lưu mô hình mạng đã huấn luyện. |

### 5.3. Các chương trình mô phỏng và so sánh

| Tệp | Chức năng |
|---|---|
| `Figure2.m` | Chương trình vẽ hình theo kịch bản nghiên cứu được định nghĩa trong tệp. |
| `hinh3.m` | So sánh OP giữa phương pháp Deep Learning, Block-Correlation và mô phỏng Jakes theo ngưỡng SNR/SIR. |
| `hinh4.m` | So sánh OP theo số lượng cổng và số người dùng trong các phương pháp được triển khai. |
| `hinh5.m` | Khảo sát thời gian tính toán giữa Block-Correlation và Deep Learning. |
| `sosanhComplexity.m` | So sánh thời gian tính toán của phương pháp BC và ML. |
| `sosanhOP_SIIR.m` | Mô phỏng so sánh OP theo ngưỡng SIR trong cấu hình được thiết lập. |
| `sosanhOP_fix.m` | So sánh OP giữa các phương pháp trong kịch bản tham số cố định. |
| `fig_so_sanh_complexity.png` | Hình ảnh phục vụ so sánh độ phức tạp tính toán. |
| `so_sanh_Pout.png` | Hình ảnh phục vụ so sánh xác suất gián đoạn. |

Tên tệp và mô tả trên được xây dựng từ cấu trúc repository và các phần mã nguồn đã kiểm tra. Đối với những kịch bản cụ thể, hãy xem trực tiếp phần tham số ở đầu mỗi tệp MATLAB.

## 6. Công nghệ sử dụng

- **MATLAB:** Ngôn ngữ chính để mô hình hóa, tính toán số và mô phỏng hệ thống.
- **Deep Learning Toolbox:** Sử dụng các lớp mạng nơ-ron và hàm `trainNetwork`, `trainingOptions`, `predict`.
- **Mô phỏng Monte Carlo:** Đánh giá xác suất gián đoạn dựa trên nhiều mẫu kênh truyền ngẫu nhiên.
- **Mô hình Jakes:** Mô tả tương quan không gian giữa các cổng anten.
- **Tích phân số và Quadrature:** Tính toán OP theo các biểu thức lý thuyết.
- **Tệp MAT:** Lưu trữ tập dữ liệu và mô hình Deep Learning.

Tùy phiên bản MATLAB và các hàm thực tế được gọi, người dùng có thể cần cài đặt thêm toolbox phù hợp.

## 7. Hướng dẫn cài đặt và chạy

### 7.1. Chuẩn bị môi trường

1. Cài đặt MATLAB.
2. Cài đặt Deep Learning Toolbox nếu chưa có.
3. Tải repository về máy:

```bash
git clone https://github.com/duongquocbao222/Scientific_Research.git
```

4. Mở MATLAB và chuyển Current Folder đến thư mục dự án:

```matlab
cd('duong_dan_den_Scientific_Research/NCKH_FAS_BCvsML-main')
```

Thay `duong_dan_den_Scientific_Research` bằng đường dẫn thực tế trên máy.

### 7.2. Thêm thư mục Core vào MATLAB path

```matlab
addpath('Core/');
```

Thao tác này cho phép MATLAB tìm thấy các hàm như `CalcOutage`, `SimOutage` và `BlockCorrelation`.

### 7.3. Tạo tập dữ liệu

Sau khi kiểm tra các hàm và điều kiện chạy, có thể tạo tập dữ liệu bằng:

```matlab
dataset_generate(100);
```

Tham số `100` là số mẫu cần sinh trong ví dụ. Giá trị này chỉ phù hợp để thử nghiệm ban đầu, không phải kích thước tập dữ liệu đã được xác nhận cho kết quả nghiên cứu cuối cùng.

Hàm `dataset_generate` thực hiện các công việc chính:

1. Sinh tham số hệ thống ngẫu nhiên.
2. Tạo các phương án phân chia cổng anten.
3. Tính OP cho từng phương án.
4. Lưu cấu hình có OP thấp nhất trong số các phương án được xét.
5. Lưu dữ liệu vào `dataset_final.mat`.

**Lưu ý:** Hàm `GenerateAllL.m` sử dụng cách liệt kê các vector phân chia số nguyên. Số phương án có thể tăng rất nhanh khi \(N\) lớn và số khối \(B\) tăng. Nên bắt đầu với bài toán nhỏ và kiểm tra thời gian chạy trước khi sinh dữ liệu quy mô lớn.

### 7.4. Huấn luyện mô hình Deep Learning

Sau khi đã có tập dữ liệu hợp lệ, chạy:

```matlab
train_DL_softmax;
```

Chương trình sẽ:

1. Đọc `dataset_final.mat`.
2. Tách đầu vào và đầu ra.
3. Chuẩn hóa đầu ra theo \(N\).
4. Xáo trộn và chia dữ liệu thành tập huấn luyện và tập kiểm tra.
5. Huấn luyện mạng nơ-ron.
6. Lưu mô hình vào `trained_net_softmax.mat`.

Cấu hình được khai báo trong mã hiện tại gồm 50 epoch, mini-batch size 256 và learning rate ban đầu \(10^{-3}\).

Sau khi huấn luyện, cần đánh giá mô hình trên dữ liệu chưa dùng để huấn luyện và kiểm tra khả năng tổng quát hóa.

### 7.5. Chạy mô phỏng so sánh

Sau khi xác nhận mô hình đã được huấn luyện và lưu hợp lệ, có thể chạy các chương trình so sánh như:

```matlab
hinh3;
```

```matlab
hinh4;
```

```matlab
hinh5;
```

Các chương trình này khảo sát những khía cạnh khác nhau:

- `hinh3.m`: So sánh OP theo ngưỡng SNR/SIR.
- `hinh4.m`: So sánh OP theo số lượng cổng anten và số người dùng.
- `hinh5.m`: So sánh thời gian tính toán.

Mỗi chương trình có cấu hình tham số riêng. Hãy kiểm tra các biến đầu vào, số vòng lặp và điều kiện hậu xử lý trước khi chạy mô phỏng thời gian dài.

## 8. Các chỉ số đánh giá

### 8.1. Outage Probability (OP)

Xác suất gián đoạn biểu thị khả năng chất lượng tín hiệu không đạt ngưỡng yêu cầu.

Trong trường hợp nhiều người dùng, OP được đánh giá theo ngưỡng SIR; trong trường hợp một người dùng, hàm tính toán sử dụng ngưỡng SNR.

OP thấp hơn thường biểu thị hiệu năng tốt hơn trong cùng điều kiện so sánh.

### 8.2. So sánh với mô hình Jakes

Mô phỏng dựa trên ma trận tương quan Jakes cung cấp một phương pháp tham chiếu để kiểm tra độ chính xác của các phương pháp xấp xỉ.

Cần bảo đảm các phương pháp được so sánh sử dụng cùng tham số hệ thống, ngưỡng đánh giá và quy ước đơn vị.

### 8.3. Thời gian tính toán

Thời gian chạy được đo bằng `tic` và `toc` trong MATLAB.

Khi so sánh, nên thực hiện nhiều lần, sử dụng cùng môi trường phần cứng và phân biệt rõ:

- Thời gian xây dựng ma trận tương quan.
- Thời gian tính toán cấu hình phân chia.
- Thời gian dự đoán của mạng đã huấn luyện.
- Thời gian đánh giá OP.

Cần tránh kết luận về tốc độ tổng thể chỉ dựa trên thời gian `predict` nếu các chi phí tiền xử lý và hậu xử lý khác nhau giữa các phương pháp.

## 9. Hạn chế và lưu ý kỹ thuật

Một số điểm cần kiểm tra trước khi sử dụng kết quả làm kết quả nghiên cứu chính thức:

1. **Tập dữ liệu:** Kiểm tra `dataset_final.mat` có chứa biến `final_dataset` với đủ số mẫu hay không. Nếu tệp chưa có dữ liệu hợp lệ, cần tạo lại.
2. **Tính nhất quán đơn vị:** Một số chương trình sử dụng ngưỡng ở thang tuyến tính, trong khi biến đầu vào hoặc chú thích có thể biểu diễn theo dB. Cần thống nhất cách chuyển đổi trước khi huấn luyện và đánh giá.
3. **Ràng buộc phân chia:** Sau khi dự đoán và làm tròn, phải kiểm tra mọi \(L_b\) hợp lệ và tổng các phần tử bằng \(N\).
4. **Phương pháp tạo nhãn:** Cấu hình có OP thấp nhất trong tập ứng viên chưa chắc là nghiệm tối ưu toàn cục.
5. **Chia dữ liệu:** Nên có tập kiểm tra độc lập và tránh sử dụng tập kiểm tra như tập đánh giá cuối cùng sau nhiều lần điều chỉnh mô hình.
6. **So sánh công bằng:** Các phương pháp phải dùng chung điều kiện mô phỏng và tiêu chí đánh giá.
7. **Tính tái lập:** Nên cố định seed ngẫu nhiên, lưu cấu hình tham số và ghi lại phiên bản MATLAB cùng các toolbox đã sử dụng.
8. **Hiệu năng tính toán:** Các vòng lặp tìm kiếm phương án phân chia có thể tốn nhiều thời gian khi số cổng anten và số khối tăng.

## 10. Hướng phát triển

Dự án có thể được mở rộng theo các hướng sau:

- Tối ưu cách sinh dữ liệu và tìm kiếm cấu hình phân chia cổng anten.
- Khảo sát các kiến trúc mạng Deep Learning khác nhau.
- So sánh sai số dự đoán cấu hình và chênh lệch OP so với phương pháp tham chiếu.
- Đánh giá khả năng tổng quát hóa khi thay đổi \(W\), \(N\), \(U\) và ngưỡng SIR.
- Khảo sát độ nhạy với mô hình tương quan kênh truyền.
- Chuẩn hóa quy trình đo thời gian tính toán và báo cáo kết quả thực nghiệm.
- Xây dựng quy trình chạy mô phỏng tự động, lưu dữ liệu và tái tạo các hình trong báo cáo nghiên cứu.

## 11. Tác giả và liên kết

- **GitHub:** [duongquocbao222/Scientific_Research](https://github.com/duongquocbao222/Scientific_Research)
- **Ngôn ngữ chính:** MATLAB
- **Lĩnh vực:** Truyền thông không dây, Fluid Antenna System, FAMA, mô hình hóa kênh truyền và Deep Learning.

## 12. Kết luận

Dự án nghiên cứu việc kết hợp mô hình tương quan khối với Deep Learning để dự đoán cấu hình phân chia cổng anten trong hệ thống FAMA. Thông qua mô phỏng OP và đo thời gian tính toán, dự án hướng đến đánh giá sự cân bằng giữa hiệu năng truyền thông và chi phí tính toán.

README này mô tả mục tiêu, cơ sở lý thuyết, cấu trúc mã nguồn và quy trình chạy chương trình. Các kết quả định lượng và kết luận về mức độ cải thiện hiệu năng cần được xác nhận bằng các thí nghiệm có thể tái lập từ mã nguồn và dữ liệu hợp lệ.
