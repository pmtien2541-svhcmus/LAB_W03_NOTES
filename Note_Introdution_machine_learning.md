# Cheat Sheet: Học Pandas & Machine Learning

Ví dụ xuyên suốt: dự đoán giá nhà ở Melbourne bằng `scikit-learn`.

---

## 1. Đọc dữ liệu với Pandas

```python
import pandas as pd

melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'  # Đường dẫn file
melbourne_data = pd.read_csv(melbourne_file_path)  # Đọc file CSV
melbourne_data.describe()  # Thống kê dữ liệu
melbourne_data.columns     # Liệt kê các cột của DataFrame
```

**Ý nghĩa các chỉ số trong `describe()`:**

| Chỉ số | Ý nghĩa |
|---|---|
| `count` | Số lượng giá trị |
| `mean` | Giá trị trung bình |
| `std` | Độ lệch chuẩn, cho biết dữ liệu phân tán nhiều hay ít |
| `min` | Giá trị nhỏ nhất |
| `max` | Giá trị lớn nhất |
| `25%`, `50%`, `75%` | Tứ phân vị |

---

## 2. Chọn Prediction Target & Features

### Prediction Target (đầu ra cần dự đoán)

```python
y = melbourne_data['Price']  # Dự đoán giá nhà -> lấy cột Price
```

### Features (đặc trưng đầu vào)

Thông tin mô hình dùng để dự đoán. Ví dụ: số phòng, khoảng cách đến trung tâm, diện tích đất.

```python
features = ['Rooms', 'Distance', 'Landsize']
X = melbourne_data[features]
```

> **Lưu ý cách lấy cột:**
> ```python
> df['Price']               # Một cột -> thường trả về Series
> df[['Rooms', 'Distance']] # Nhiều cột -> trả về DataFrame
> ```

### Review dữ liệu trước khi build model

```python
print(X.describe())  # Thống kê
print(X.head())      # In 5 dòng đầu tiên
print(X)             # In toàn bộ DataFrame
print(y.head())
```

- `X` (Rooms, Distance, Landsize) → vai trò **đầu vào**
- `y` (Price) → **đầu ra** cần dự đoán

---

## 3. Xây dựng model & dự đoán

```python
from sklearn.tree import DecisionTreeRegressor

melbourne_model = DecisionTreeRegressor(random_state=1)
melbourne_model.fit(X, y)  # Huấn luyện

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))
```

---

## 4. Model Validation (Kiểm chứng mô hình)

Xây xong model chưa chắc đã tốt. Kiểm chứng là tiền đề để cải thiện model qua các vòng lặp thử nghiệm.

### MAE (Mean Absolute Error - Sai số tuyệt đối trung bình)

Ví dụ: nhà giá 150,000$ mà dự đoán 100,000$ → sai số = 150,000 - 100,000 = **50,000$**.

MAE là thước đo chất lượng mô hình.

```python
from sklearn.metrics import mean_absolute_error

predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices)
```

### Vấn đề: đánh giá trên dữ liệu cũ

Model đo trên chính dữ liệu đã huấn luyện sẽ cho kết quả rất chính xác, nhưng quy luật đó **không còn đúng với dữ liệu mới** → độ chính xác thấp.

**Giải pháp:** Tách một phần dữ liệu ra khỏi quá trình huấn luyện, dùng phần đó để kiểm tra → gọi là **Validation Data**.

### `train_test_split`

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Tách dữ liệu thành train và validation, cho cả features và target
# random_state cố định giúp mỗi lần chạy đều chia giống nhau
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
# Mặc định: 75% train, 25% validation

# Tuỳ chỉnh tỉ lệ:
# train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=1)
# test_size=0.2 -> validation 20%, training 80%

melbourne_model = DecisionTreeRegressor()      # Định nghĩa model
melbourne_model.fit(train_X, train_y)          # Huấn luyện trên tập train

val_predictions = melbourne_model.predict(val_X)  # Dự đoán trên tập validation
print(mean_absolute_error(val_y, val_predictions))
```

---

## 5. Underfitting & Overfitting

- **Underfitting:** Mô hình học chưa đủ, dự đoán kém → số node **ít**.
- **Overfitting:** Mô hình học quá sát dữ liệu huấn luyện (kể cả chi tiết riêng biệt hoặc nhiễu), nên dự đoán kém trên dữ liệu mới → số node **nhiều**.

**Cách chẩn đoán:**

| MAE Training | MAE Validation | Kết luận |
|---|---|---|
| Cao | Cao | Underfitting |
| Thấp | Cao | Overfitting |
| Thấp | Thấp | Tốt (GOAT) |

### Dùng `max_leaf_nodes` để cân bằng

`max_leaf_nodes` kiểm soát số node lá, giúp tránh cả hai trường hợp trên.

```python
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae
```

### So sánh MAE với nhiều giá trị `max_leaf_nodes`

```python
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]

save_max_leaf_nodes = {}
for max_leaf_nodes in candidate_max_leaf_nodes:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    save_max_leaf_nodes[max_leaf_nodes] = my_mae

# Giá trị max_leaf_nodes cho MAE nhỏ nhất
best_tree_size = min(save_max_leaf_nodes, key=save_max_leaf_nodes.get)
```

### Model cuối cùng

Khi đã tìm được giá trị tốt nhất (ví dụ 50), huấn luyện lại bằng **toàn bộ dữ liệu**:

```python
final_model = DecisionTreeRegressor(max_leaf_nodes=50, random_state=1)
final_model.fit(X, y)
```

---

## 6. Random Forest

Decision Tree không quá tinh vi. Random Forest (nhiều cây kết hợp) tránh underfitting và overfitting hiệu quả hơn.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))
```

---

## Quy trình tổng quát

1. Đọc dữ liệu (`pd.read_csv`) → khám phá (`describe`, `columns`, `head`)
2. Chọn target `y` và features `X`
3. Chia dữ liệu: `train_test_split`
4. Khởi tạo model → `fit` → `predict`
5. Đánh giá bằng MAE trên validation
6. Điều chỉnh (`max_leaf_nodes`) để cân bằng under/overfitting
7. Thử model mạnh hơn (Random Forest)
8. Huấn luyện model cuối trên toàn bộ dữ liệu
