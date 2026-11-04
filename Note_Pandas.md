# Tổng hợp kiến thức Pandas & Machine Learning đã học

> Ghi chú học tập từ Kaggle Learn: **Introduction to Machine
> Learning**.\
> Ví dụ chính sử dụng bộ dữ liệu Melbourne Housing để dự đoán giá nhà.

## 1. Vai trò của Pandas và Scikit-learn

-   **Pandas**: đọc, xem, chọn và xử lý dữ liệu dạng bảng.
-   **Scikit-learn (`sklearn`)**: cung cấp các thuật toán Machine
    Learning, công cụ chia dữ liệu và cách đánh giá mô hình.
-   **Machine Learning**: cho mô hình học mối quan hệ từ dữ liệu để dự
    đoán kết quả cho dữ liệu mới.

Quy trình cơ bản:

1.  Đọc và khám phá dữ liệu.
2.  Chọn giá trị cần dự đoán (`y`) và các đặc trưng đầu vào (`X`).
3.  Chia dữ liệu thành tập huấn luyện và tập kiểm chứng.
4.  Huấn luyện mô hình bằng `.fit()`.
5.  Dự đoán bằng `.predict()`.
6.  Đánh giá bằng MAE.
7.  Thử điều chỉnh mô hình và chọn mô hình phù hợp.
8.  Khi đã lựa chọn xong, có thể huấn luyện mô hình cuối cùng trên toàn
    bộ dữ liệu có nhãn.

------------------------------------------------------------------------

## 2. Đọc dữ liệu bằng Pandas

``` python
import pandas as pd

melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
```

-   `import pandas as pd`: nhập thư viện Pandas với tên viết tắt `pd`.
-   `pd.read_csv(path)`: đọc file CSV và trả về một `DataFrame`.
-   CSV là file dữ liệu dạng bảng; không nhất thiết là file Excel
    (`.xlsx`).

### Khám phá dữ liệu

``` python
melbourne_data.head()       # Mặc định xem 5 dòng đầu
melbourne_data.shape        # (số dòng, số cột)
melbourne_data.columns      # Tên các cột
melbourne_data.describe()   # Thống kê các cột số
melbourne_data.info()       # Kiểu dữ liệu và số giá trị không thiếu
```

Trong kết quả `describe()`:

-   `count`: số giá trị không bị thiếu.
-   `mean`: giá trị trung bình.
-   `std`: độ lệch chuẩn, cho biết mức độ phân tán.
-   `min`, `max`: giá trị nhỏ nhất và lớn nhất.
-   `25%`, `50%`, `75%`: các tứ phân vị; `50%` là trung vị.

`print(df)` in ra dữ liệu, nhưng với bảng lớn có thể tạo đầu ra rất dài.
Thường nên dùng `head()` để xem nhanh.

------------------------------------------------------------------------

## 3. Chọn target (`y`) và features (`X`)

Giả sử muốn dự đoán giá nhà ở Melbourne:

``` python
y = melbourne_data['Price']

features = ['Rooms', 'Distance', 'Landsize']
X = melbourne_data[features]
```

-   **Target (`y`)**: giá trị mà mô hình cần dự đoán. Ở đây là `Price`.
-   **Features (`X`)**: các thông tin đầu vào dùng để dự đoán. Ở đây là
    `Rooms`, `Distance`, `Landsize`.
-   `features`: danh sách tên các cột được chọn.

### Khác nhau giữa chọn một cột và nhiều cột

``` python
df['Price']                   # Một cột, thường trả về Series
df[['Rooms', 'Distance']]     # Nhiều cột, trả về DataFrame
```

Nên kiểm tra dữ liệu đã chọn:

``` python
print(X.head())
print(y.head())
print(X.describe())
```

Mỗi hàng trong `X` phải tương ứng với giá trị target ở cùng hàng trong
`y`.

------------------------------------------------------------------------

## 4. Huấn luyện Decision Tree

``` python
from sklearn.tree import DecisionTreeRegressor

melbourne_model = DecisionTreeRegressor(random_state=1)
melbourne_model.fit(X, y)
```

-   `DecisionTreeRegressor`: cây quyết định dùng cho bài toán hồi quy,
    tức dự đoán giá trị số.
-   `random_state=1`: cố định yếu tố ngẫu nhiên để kết quả có thể tái
    lập trong cùng điều kiện.
-   `.fit(X, y)`: huấn luyện mô hình từ features và target.

### Dự đoán

``` python
predictions = melbourne_model.predict(X.head())
print(predictions)
```

`.predict(data)` dùng mô hình đã huấn luyện để dự đoán kết quả cho dữ
liệu đầu vào. Dữ liệu đưa vào cần có các cột phù hợp với những cột mô
hình đã được huấn luyện.

**Lưu ý:** Dự đoán trên chính dữ liệu đã dùng để huấn luyện không cho
biết chắc mô hình sẽ làm tốt trên dữ liệu mới.

------------------------------------------------------------------------

## 5. Đánh giá mô hình bằng MAE

MAE là viết tắt của **Mean Absolute Error** --- sai số tuyệt đối trung
bình.

Công thức:

\[ MAE =
`\frac{1}{n}`{=tex}`\sum`{=tex}\_{i=1}\^{n}\|y_i-`\hat{y}`{=tex}\_i\| \]

Trong đó:

-   (y_i): giá trị thật.
-   (`\hat{y}`{=tex}\_i): giá trị mô hình dự đoán.
-   (n): số mẫu được đánh giá.

Ví dụ: giá thật là 150.000, mô hình dự đoán 100.000 thì sai số tuyệt đối
của mẫu đó là 50.000. MAE là trung bình sai số tuyệt đối của tất cả các
mẫu.

``` python
from sklearn.metrics import mean_absolute_error

predicted_home_prices = melbourne_model.predict(X)
mae = mean_absolute_error(y, predicted_home_prices)
print(mae)
```

MAE càng thấp thường càng tốt khi so sánh các mô hình trên **cùng tập
đánh giá và cùng target**. Tuy nhiên, MAE trên dữ liệu huấn luyện có thể
quá lạc quan vì mô hình đã học từ chính dữ liệu đó.

------------------------------------------------------------------------

## 6. Chia dữ liệu thành training và validation

Để kiểm tra khả năng dự đoán dữ liệu chưa dùng để huấn luyện, ta chia dữ
liệu thành hai phần:

-   **Training set**: dùng để huấn luyện mô hình.
-   **Validation set**: dùng để đánh giá mô hình và so sánh lựa chọn mô
    hình.

``` python
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(
    X, y, random_state=0
)
```

Theo mặc định, `train_test_split()` thường dành 75% dữ liệu cho training
và 25% cho validation.

Muốn dùng 80% training và 20% validation:

``` python
train_X, val_X, train_y, val_y = train_test_split(
    X, y, test_size=0.2, random_state=1
)
```

-   `test_size=0.2`: dành 20% làm tập validation.
-   `random_state=1`: giữ cách chia ổn định giữa các lần chạy.
-   `random_state` không quyết định tỷ lệ chia; `test_size` mới quyết
    định tỷ lệ.

### Huấn luyện và đánh giá trên validation

``` python
model = DecisionTreeRegressor(random_state=0)
model.fit(train_X, train_y)

val_predictions = model.predict(val_X)
val_mae = mean_absolute_error(val_y, val_predictions)
print(val_mae)
```

Không dùng `val_y` để huấn luyện khi đang đánh giá trên validation; nếu
dùng, kết quả đánh giá sẽ không còn phản ánh đúng khả năng dự đoán trên
dữ liệu chưa thấy.

------------------------------------------------------------------------

## 7. Underfitting và Overfitting

### Underfitting --- học chưa đủ

Mô hình quá đơn giản nên không nắm bắt được các mối quan hệ quan trọng
trong dữ liệu. Nó có thể dự đoán kém cả trên training lẫn validation.

### Overfitting --- học quá sát dữ liệu huấn luyện

Mô hình học cả những chi tiết đặc thù hoặc nhiễu trong training, dẫn đến
kết quả rất tốt trên training nhưng kém hơn trên dữ liệu mới.

Cách đọc MAE một cách khái quát:

  MAE training   MAE validation   Khả năng có thể xảy ra
  -------------- ---------------- ---------------------------
  Cao            Cao              Có thể underfitting
  Thấp           Cao hơn nhiều    Có thể overfitting
  Thấp           Thấp             Mô hình có vẻ phù hợp tốt

Đây là dấu hiệu để chẩn đoán, không phải quy tắc tuyệt đối. Cần xem xét
mức độ chênh lệch và bối cảnh dữ liệu.

------------------------------------------------------------------------

## 8. Điều chỉnh độ phức tạp của Decision Tree

`max_leaf_nodes` giới hạn số nút lá của cây:

-   Giới hạn nhỏ hơn thường tạo cây đơn giản hơn.
-   Giới hạn lớn hơn cho phép cây học nhiều chi tiết hơn, nhưng có thể
    tăng nguy cơ overfitting.
-   Không có một giá trị luôn tốt nhất cho mọi bộ dữ liệu; hãy chọn dựa
    trên MAE validation.

Tạo hàm tính MAE:

``` python
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(
        max_leaf_nodes=max_leaf_nodes,
        random_state=0
    )
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    return mean_absolute_error(val_y, preds_val)
```

Thử nhiều giá trị:

``` python
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
save_max_leaf_nodes = {}

for max_leaf_nodes in candidate_max_leaf_nodes:
    my_mae = get_mae(
        max_leaf_nodes, train_X, val_X, train_y, val_y
    )
    save_max_leaf_nodes[max_leaf_nodes] = my_mae

best_tree_size = min(save_max_leaf_nodes, key=save_max_leaf_nodes.get)

print("MAE theo số leaf nodes:", save_max_leaf_nodes)
print("Số leaf nodes tốt nhất:", best_tree_size)
```

`min(dictionary, key=dictionary.get)` trả về **khóa** có giá trị nhỏ
nhất. Trong ví dụ này, khóa là số leaf nodes có MAE thấp nhất trong danh
sách đã thử.

### Huấn luyện mô hình cuối cùng

Sau khi chọn cấu hình bằng validation, có thể huấn luyện lại mô hình
cuối cùng trên toàn bộ `X` và `y`:

``` python
final_model = DecisionTreeRegressor(
    max_leaf_nodes=best_tree_size,
    random_state=0
)
final_model.fit(X, y)
```

Nên dùng `best_tree_size` thay vì ghi cứng `50`, vì kết quả tốt nhất phụ
thuộc vào MAE thực tế của các giá trị đã thử.

------------------------------------------------------------------------

## 9. Random Forest Regressor

Random Forest kết hợp nhiều cây quyết định. Với bài toán hồi quy, các
cây đưa ra dự đoán và mô hình thường kết hợp bằng cách lấy trung bình.
Cách làm này thường ổn định hơn một cây đơn lẻ, nhưng không bảo đảm luôn
tốt hơn.

### Huấn luyện và đánh giá

``` python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)

melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))
```

-   `RandomForestRegressor`: mô hình Random Forest cho bài toán hồi quy.
-   `.fit(train_X, train_y)`: huấn luyện bằng tập training.
-   `.predict(val_X)`: dự đoán trên tập validation.
-   `mean_absolute_error(val_y, melb_preds)`: đo sai số dự đoán trên
    validation.

Có thể đặt số lượng cây bằng `n_estimators`, ví dụ:

``` python
forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=1
)
```

`n_estimators=100` nghĩa là sử dụng 100 cây. Nếu không chỉ định,
Scikit-learn sẽ dùng giá trị mặc định của phiên bản đang cài.

### Mô hình cuối cùng dùng toàn bộ dữ liệu

Sau khi đánh giá và lựa chọn mô hình, có thể huấn luyện mô hình cuối
cùng bằng tất cả dữ liệu có nhãn:

``` python
rf_model_on_full_data = RandomForestRegressor(random_state=1)
rf_model_on_full_data.fit(X, y)
```

-   `fit(train_X, train_y)`: dùng phần training sau khi chia.
-   `fit(X, y)`: dùng toàn bộ dữ liệu ban đầu có nhãn.

Hãy giữ lại validation trong giai đoạn lựa chọn mô hình; chỉ dùng toàn
bộ dữ liệu khi đã hoàn tất việc đánh giá/lựa chọn.

------------------------------------------------------------------------

## 10. Dự đoán cho dữ liệu test

Trong bài Kaggle, file `test.csv` thường không có cột target (chẳng hạn
`SalePrice`), vì đó chính là giá trị cần dự đoán.

Ví dụ tổng quát:

``` python
test_data_path = '../input/test.csv'
test_data = pd.read_csv(test_data_path)

test_X = test_data[features]
test_preds = rf_model_on_full_data.predict(test_X)
```

Ý nghĩa:

-   `test_data`: dữ liệu test đọc từ file.
-   `test_X = test_data[features]`: chọn đúng những cột đã dùng khi huấn
    luyện.
-   `test_preds`: các giá trị mô hình dự đoán.

**Quan trọng:** `features` phải giống danh sách cột đã dùng để tạo `X`,
và mô hình phải được `.fit()` trước khi gọi `.predict()`.

Đường dẫn file phụ thuộc môi trường. Trong Kaggle, có thể cần đường dẫn
đầy đủ như:

``` python
test_data_path = '/kaggle/input/home-data-for-ml-course/test.csv'
```

Nếu gặp `FileNotFoundError`, kiểm tra lại tên thư mục và file. Nếu gặp
`NameError`, kiểm tra xem biến đã được tạo và cell tương ứng đã chạy
chưa. Nếu gặp `NotFittedError`, hãy gọi `.fit()` trên mô hình hiện tại
trước `.predict()`.

### Tạo file submission (ví dụ bài dự đoán giá nhà)

``` python
output = pd.DataFrame({
    'Id': test_data.Id,
    'SalePrice': test_preds
})

output.to_csv('submission.csv', index=False)
```

Chỉ dùng tên cột `Id` và `SalePrice` nếu đúng với yêu cầu của cuộc thi
đang làm.

------------------------------------------------------------------------

## 11. Những lỗi thường gặp đã gặp trong quá trình học

  ---------------------------------------------------------------------------------------------
  Lỗi                                           Nguyên nhân thường gặp  Cách kiểm tra/sửa
  --------------------------------------------- ----------------------- -----------------------
  `NameError: name 'X' is not defined`          Chưa tạo `X` hoặc chưa  Tạo lại `X` từ
                                                chạy cell tạo biến      DataFrame training và
                                                                        danh sách `features`

  `NameError: name 'features' is not defined`   Chưa khai báo danh sách Khai báo `features`
                                                cột                     giống lúc tạo `X`

  `AttributeError` với `test_data.SalePrice`    File test không có cột  Không tạo `y` từ file
                                                target                  test; lấy `y` từ file
                                                                        training

  `FileNotFoundError`                           Đường dẫn sai hoặc file Kiểm tra tên file, thư
                                                không tồn tại           mục và đường dẫn

  `NotFittedError`                              Mô hình hiện tại chưa   Gọi `.fit(X, y)` trước
                                                được huấn luyện         `.predict()`

  Lỗi cột khi dự đoán                           `test_X` có cột khác    Dùng
                                                với dữ liệu huấn luyện  `test_data[features]`
                                                                        với đúng danh sách cột

  `OSError: Read-only file system`              Cố ghi/tạo symbolic     Đọc file bằng đường dẫn
                                                link trong thư mục chỉ  hiện có hoặc tạo link ở
                                                đọc                     thư mục được phép ghi
  ---------------------------------------------------------------------------------------------

Trong Jupyter/Kaggle Notebook, nếu kernel bị khởi động lại, các biến
trong bộ nhớ có thể mất. Khi đó cần chạy lại các cell theo thứ tự từ
trên xuống.

------------------------------------------------------------------------

## 12. Cheat sheet ngắn

``` python
# 1. Đọc dữ liệu
import pandas as pd
data = pd.read_csv('data.csv')

# 2. Chọn target và features
y = data['Target']
features = ['Feature1', 'Feature2']
X = data[features]

# 3. Chia dữ liệu
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# 4. Huấn luyện
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state=1)
model.fit(train_X, train_y)

# 5. Dự đoán và đánh giá
from sklearn.metrics import mean_absolute_error
val_preds = model.predict(val_X)
print(mean_absolute_error(val_y, val_preds))

# 6. Mô hình cuối cùng trên toàn bộ dữ liệu
from sklearn.ensemble import RandomForestRegressor
final_model = RandomForestRegressor(random_state=1)
final_model.fit(X, y)

# 7. Dự đoán dữ liệu mới
test_data = pd.read_csv('test.csv')
test_X = test_data[features]
test_preds = final_model.predict(test_X)
```

Trong ví dụ trên, hãy thay `Target`, `Feature1`, `Feature2` và đường dẫn
file bằng tên cột/đường dẫn thực tế của bộ dữ liệu.

------------------------------------------------------------------------

## 13. Những điều cần nhớ

1.  `X` là dữ liệu đầu vào; `y` là kết quả cần dự đoán.
2.  `.fit()` dùng để huấn luyện; `.predict()` dùng để dự đoán.
3.  MAE thấp hơn thường tốt hơn khi so sánh trên cùng tập đánh giá.
4.  Cần đánh giá bằng dữ liệu chưa dùng để huấn luyện, chẳng hạn
    validation.
5.  Decision Tree có thể underfit hoặc overfit; `max_leaf_nodes` giúp
    điều chỉnh độ phức tạp.
6.  Random Forest kết hợp nhiều cây và thường cho kết quả ổn định hơn
    một cây đơn.
7.  Khi dự đoán test, phải chọn đúng các cột `features` đã dùng lúc huấn
    luyện.
8.  Dữ liệu test thường không có target; không lấy `y` từ file test.
9.  Tên biến và đường dẫn phải nhất quán; chạy cell theo đúng thứ tự.
10. Không có mô hình tốt nhất cho mọi bộ dữ liệu: hãy đánh giá bằng
    validation để lựa chọn.

**Bước học tiếp theo phù hợp:** hoàn thành bài Introduction to Machine
Learning, sau đó học Intermediate Machine Learning về giá trị thiếu, dữ
liệu phân loại, pipeline, cross-validation và data leakage.
